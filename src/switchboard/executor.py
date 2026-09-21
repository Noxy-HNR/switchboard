"""Carrying out a model switch.

On this machine a "switch" is a restart: one llama-server, one resident model,
12 GB of VRAM. Switchboard does not restart it directly. It asks the dashboard
(``C:\\AI\\dashboard-server.ps1``, port 9090) to do it through the same
``/api/models/load`` endpoint the dashboard's own Models tab and
``model-manage.ps1`` use, so there is exactly one code path that starts
llama-server on this machine and Switchboard is not a second one.

Consequences that callers must know about, and that are deliberate:

* the dashboard also writes ``activeModel`` into ``C:\\AI\\Config\\models.json``,
  so a switch changes which model ``start.ps1`` will load next time. That file
  is backed up here before the first switch of a process.
* the restart drops whatever llama-server was doing. The router checks
  ``/slots`` and its own in-flight requests before asking for a switch.
* the load itself is measured, and the measurement is what later routing uses
  to decide whether a switch is worth it.
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import httpx

from switchboard.config import Config, PROJECT_ROOT
from switchboard.registry import ModelEntry
from switchboard.upstream import LlamaClient


@dataclass(slots=True)
class SwitchResult:
    ok: bool
    executor: str
    from_model: str | None
    to_model: str
    load_ms: float | None = None
    error: str | None = None
    # True when nothing was done because the model was already resident.
    noop: bool = False

    def describe(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "executor": self.executor,
            "from_model": self.from_model,
            "to_model": self.to_model,
            "load_ms": None if self.load_ms is None else round(self.load_ms, 1),
            "error": self.error,
            "noop": self.noop,
        }


class SwitchExecutor(Protocol):
    name: str

    async def available(self) -> tuple[bool, str]: ...

    async def switch(self, target: ModelEntry, *, from_model: str | None) -> SwitchResult: ...


class NullExecutor:
    """Refuses every switch. Used in shadow mode and when routing is off."""

    name = "none"

    def __init__(self, reason: str = "switching is disabled in this mode") -> None:
        self._reason = reason

    async def available(self) -> tuple[bool, str]:
        return False, self._reason

    async def switch(self, target: ModelEntry, *, from_model: str | None) -> SwitchResult:
        return SwitchResult(
            ok=False,
            executor=self.name,
            from_model=from_model,
            to_model=target.id,
            error=self._reason,
        )


class DashboardExecutor:
    """Switches by asking the C:\\AI dashboard to load a model."""

    name = "dashboard"

    def __init__(
        self,
        config: Config,
        upstream: LlamaClient,
        *,
        client: httpx.AsyncClient | None = None,
        backup_dir: Path | None = None,
    ) -> None:
        self._config = config
        self._upstream = upstream
        self._base_url = config.stack.dashboard_url.rstrip("/")
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=self._base_url, timeout=httpx.Timeout(30.0, connect=3.0)
        )
        self._backup_dir = backup_dir or (PROJECT_ROOT / "backups")
        self._backed_up = False

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def available(self) -> tuple[bool, str]:
        """Whether the dashboard is reachable and can load models."""
        try:
            response = await self._client.get("/api/models", timeout=httpx.Timeout(4.0, connect=2.0))
        except httpx.HTTPError as error:
            return False, f"dashboard unreachable at {self._base_url} ({type(error).__name__})"
        if response.status_code != 200:
            return False, f"dashboard /api/models returned {response.status_code}"
        return True, f"dashboard reachable at {self._base_url}"

    def backup_models_config(self) -> Path | None:
        """Copy Config\\models.json aside before the dashboard rewrites it.

        The dashboard sets ``activeModel`` on every load, which changes what
        ``start.ps1`` launches next time. One backup per process is enough: the
        first one holds the state from before Switchboard touched anything.
        """
        source = self._config.stack.models_config
        if self._backed_up or not source.is_file():
            return None
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        self._backup_dir.mkdir(parents=True, exist_ok=True)
        destination = self._backup_dir / f"models.json.{stamp}.bak"
        shutil.copy2(source, destination)
        self._backed_up = True
        return destination

    async def switch(self, target: ModelEntry, *, from_model: str | None) -> SwitchResult:
        """Load `target`, then wait for llama-server to come back with it."""
        if not target.exists:
            return SwitchResult(
                ok=False,
                executor=self.name,
                from_model=from_model,
                to_model=target.id,
                error=f"{target.filename} is not present in {target.path.parent}",
            )

        self.backup_models_config()
        started = time.perf_counter()

        try:
            response = await self._client.post(
                "/api/models/load", json={"filename": target.filename}
            )
        except httpx.HTTPError as error:
            return SwitchResult(
                ok=False,
                executor=self.name,
                from_model=from_model,
                to_model=target.id,
                error=f"dashboard load request failed: {type(error).__name__}: {error}",
            )

        if response.status_code != 200:
            return SwitchResult(
                ok=False,
                executor=self.name,
                from_model=from_model,
                to_model=target.id,
                error=f"dashboard returned {response.status_code}",
            )

        try:
            payload = response.json()
        except ValueError:
            payload = {}
        if isinstance(payload, dict) and payload.get("success") is False:
            return SwitchResult(
                ok=False,
                executor=self.name,
                from_model=from_model,
                to_model=target.id,
                error=str(payload.get("message") or "dashboard refused the load"),
            )

        # The dashboard returns as soon as the process is launched; the model is
        # only usable once /health answers and /v1/models reports the new file.
        status = await self._upstream.wait_until_healthy(
            timeout_s=self._config.routing.guards.max_switch_wait_s,
            expect_model=target.filename.removesuffix(".gguf"),
        )
        load_ms = (time.perf_counter() - started) * 1000

        if not status.healthy:
            return SwitchResult(
                ok=False,
                executor=self.name,
                from_model=from_model,
                to_model=target.id,
                load_ms=load_ms,
                error=status.error or "llama-server did not come back after the load",
            )

        resident = (status.model_id or "").lower()
        if target.filename.removesuffix(".gguf").lower() not in resident:
            return SwitchResult(
                ok=False,
                executor=self.name,
                from_model=from_model,
                to_model=target.id,
                load_ms=load_ms,
                error=(
                    f"llama-server came back with {status.model_id!r}, "
                    f"not {target.filename!r}"
                ),
            )

        return SwitchResult(
            ok=True,
            executor=self.name,
            from_model=from_model,
            to_model=target.id,
            load_ms=load_ms,
        )


def build_executor(
    config: Config, upstream: LlamaClient, *, client: httpx.AsyncClient | None = None
) -> SwitchExecutor:
    """The executor implied by the configured routing mode."""
    if config.routing.mode == "active":
        return DashboardExecutor(config, upstream, client=client)
    return NullExecutor(
        reason=f"routing.mode is {config.routing.mode!r}: Switchboard reports switches instead of making them"
    )
