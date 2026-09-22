"""The llama.cpp client.

Switchboard sits in front of the llama-server that ``C:\\AI\\start.ps1`` starts
and never changes how it is launched. This module is the only place that talks
to it, and it answers the deterministic questions routing depends on:

* is it up, and which model is resident right now?
* is it busy generating for someone else? (``--parallel 1``: a second request
  queues behind the first, so "busy" changes what Switchboard may do)
* is it a plain single-model server, or a llama.cpp *router*?

Health is cached for a couple of seconds: every routed request asks, and the
answer cannot change faster than a process restart.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from switchboard.config import Config


@dataclass(slots=True)
class UpstreamStatus:
    """A point-in-time view of the llama-server."""

    healthy: bool = False
    # Whatever llama-server calls the resident model: usually the GGUF path.
    model_id: str | None = None
    model_path: str | None = None
    context_size: int | None = None
    busy: bool | None = None
    router_mode: bool = False
    latency_ms: float | None = None
    error: str | None = None
    checked_at: float = 0.0
    # Model ids a router advertises; empty for a single-model server.
    available_models: list[str] = field(default_factory=list)

    def describe(self) -> dict[str, Any]:
        return {
            "healthy": self.healthy,
            "model_id": self.model_id,
            "model_path": self.model_path,
            "context_size": self.context_size,
            "busy": self.busy,
            "router_mode": self.router_mode,
            "latency_ms": None if self.latency_ms is None else round(self.latency_ms, 1),
            "error": self.error,
            "available_models": self.available_models,
        }


class UpstreamError(RuntimeError):
    """The upstream could not be reached or answered with an error."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class LlamaClient:
    """Async client for one llama-server instance."""

    def __init__(self, config: Config, *, client: httpx.AsyncClient | None = None) -> None:
        self._config = config
        self._base_url = config.upstream.base_url.rstrip("/")
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(
                config.upstream.request_timeout_s,
                connect=config.upstream.connect_timeout_s,
            ),
            # One conversation at a time, but the decision layer and the proxy
            # can overlap, so keep a few connections alive.
            limits=httpx.Limits(max_keepalive_connections=4, max_connections=8),
        )
        self._status: UpstreamStatus | None = None

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def http(self) -> httpx.AsyncClient:
        return self._client

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    # --- state ---------------------------------------------------------

    async def status(self, *, force: bool = False) -> UpstreamStatus:
        """Current upstream state, cached for ``upstream.health_ttl_s``."""
        cached = self._status
        ttl = self._config.upstream.health_ttl_s
        if cached and not force and (time.monotonic() - cached.checked_at) < ttl:
            return cached

        started = time.perf_counter()
        status = UpstreamStatus(checked_at=time.monotonic())
        try:
            response = await self._client.get("/health", timeout=self._short_timeout())
            status.healthy = response.status_code == 200
            if not status.healthy:
                status.error = f"/health returned {response.status_code}"
        except httpx.HTTPError as error:
            status.error = f"{type(error).__name__}: {error}"
            status.latency_ms = (time.perf_counter() - started) * 1000
            self._status = status
            return status

        status.latency_ms = (time.perf_counter() - started) * 1000

        if status.healthy:
            await self._fill_model(status)
            status.busy = await self._busy()

        self._status = status
        return status

    def _short_timeout(self) -> httpx.Timeout:
        """Status calls must never inherit the long generation timeout."""
        return httpx.Timeout(4.0, connect=self._config.upstream.connect_timeout_s)

    async def _fill_model(self, status: UpstreamStatus) -> None:
        try:
            response = await self._client.get("/v1/models", timeout=self._short_timeout())
            if response.status_code == 200:
                payload = response.json()
                entries = payload.get("data") or []
                ids = [str(entry.get("id")) for entry in entries if entry.get("id")]
                status.available_models = ids
                if ids:
                    status.model_id = ids[0]
                # A router reports a status object per model and usually several
                # models; a single-model server reports exactly one, bare entry.
                status.router_mode = any(
                    isinstance(entry.get("status"), dict) for entry in entries
                )
                if status.router_mode:
                    loaded = [
                        str(entry.get("id"))
                        for entry in entries
                        if isinstance(entry.get("status"), dict)
                        and entry["status"].get("value") == "loaded"
                    ]
                    status.model_id = loaded[0] if loaded else None
        except (httpx.HTTPError, ValueError) as error:
            status.error = f"/v1/models unreadable: {type(error).__name__}: {error}"

        try:
            response = await self._client.get("/props", timeout=self._short_timeout())
            if response.status_code == 200:
                props = response.json()
                model_path = props.get("model_path") or props.get("model")
                if isinstance(model_path, str):
                    status.model_path = model_path
                    status.model_id = status.model_id or model_path
                context = props.get("n_ctx") or props.get("default_generation_settings", {}).get(
                    "n_ctx"
                )
                if isinstance(context, int):
                    status.context_size = context
        except (httpx.HTTPError, ValueError):
            # /props is optional; the model id from /v1/models is enough.
            pass

    async def _busy(self) -> bool | None:
        """Whether any slot is processing. None when the endpoint is disabled."""
        try:
            response = await self._client.get("/slots", timeout=self._short_timeout())
        except httpx.HTTPError:
            return None
        if response.status_code != 200:
            return None
        try:
            slots = response.json()
        except ValueError:
            return None
        if not isinstance(slots, list):
            return None
        for slot in slots:
            if isinstance(slot, dict) and slot.get("is_processing"):
                return True
        return False

    async def resident_model_id(self) -> str | None:
        """What llama-server has loaded right now, in one request, uncached.

        The full `status()` makes four calls. The check a request makes just
        before generating only needs this one, and it runs on every request.
        """
        try:
            response = await self._client.get("/v1/models", timeout=self._short_timeout())
        except httpx.HTTPError:
            return None
        if response.status_code != 200:
            return None
        try:
            entries = response.json().get("data") or []
        except (ValueError, AttributeError):
            return None
        if any(isinstance(entry.get("status"), dict) for entry in entries):
            loaded = [
                str(entry.get("id"))
                for entry in entries
                if isinstance(entry.get("status"), dict) and entry["status"].get("value") == "loaded"
            ]
            return loaded[0] if loaded else None
        ids = [str(entry.get("id")) for entry in entries if entry.get("id")]
        return ids[0] if ids else None

    async def wait_until_healthy(
        self, *, timeout_s: float, poll_s: float = 1.0, expect_model: str | None = None
    ) -> UpstreamStatus:
        """Poll until the server answers and (optionally) reports a given model.

        Used after a model switch: the dashboard restarts llama-server, so the
        port goes away and comes back with a different model loaded.
        """
        deadline = time.monotonic() + timeout_s
        last = await self.status(force=True)
        while time.monotonic() < deadline:
            if last.healthy and (
                expect_model is None
                or (last.model_id and expect_model.lower() in last.model_id.lower())
            ):
                return last
            await asyncio.sleep(poll_s)
            last = await self.status(force=True)
        return last

    # --- generation ----------------------------------------------------

    async def chat_completion(self, body: dict[str, Any]) -> httpx.Response:
        """Non-streaming chat completion, returned verbatim."""
        try:
            return await self._client.post("/v1/chat/completions", json=body)
        except httpx.HTTPError as error:
            raise UpstreamError(f"{type(error).__name__}: {error}") from error

    def stream_chat_completion(self, body: dict[str, Any]) -> Any:
        """Streaming chat completion as an httpx streaming context manager."""
        return self._client.stream("POST", "/v1/chat/completions", json=body)

    async def post_json(self, path: str, body: dict[str, Any], *, timeout: float | None = None) -> Any:
        """POST helper for the decision backends (completions, tokenize...)."""
        try:
            response = await self._client.post(
                path,
                json=body,
                timeout=httpx.Timeout(timeout) if timeout else None,
            )
        except httpx.HTTPError as error:
            raise UpstreamError(f"{type(error).__name__}: {error}") from error
        if response.status_code >= 400:
            raise UpstreamError(
                f"{path} returned {response.status_code}: {response.text[:200]}",
                status_code=response.status_code,
            )
        return response.json()

    async def forward(self, method: str, path: str, body: dict[str, Any] | None = None) -> httpx.Response:
        """Pass a request through untouched (used for non-chat endpoints)."""
        try:
            if method.upper() == "GET":
                return await self._client.get(path)
            return await self._client.request(method, path, json=body)
        except httpx.HTTPError as error:
            raise UpstreamError(f"{type(error).__name__}: {error}") from error
