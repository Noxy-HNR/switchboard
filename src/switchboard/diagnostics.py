"""`switchboard doctor`: find out why something is not working.

Each check answers one question, reports ok/warn/fail/skip, and where it can,
says what to do about it. Checks run against the machine, not against
assumptions: llama-server is probed, the dashboard is probed, the database is
opened, the model files are stat-ed, and the cptr config is read.

Everything here works with Switchboard stopped - that is the case where
diagnostics matter most - and nothing here changes any state.
"""

from __future__ import annotations

import asyncio
import json
import socket
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import httpx

from switchboard.config import Config, load_config
from switchboard.decisions.factory import backend_names
from switchboard.registry import build_registry
from switchboard.system import HostFacts

Status = Literal["ok", "warn", "fail", "skip"]

CPTR_CONFIG = Path.home() / ".cptr" / "config.toml"


@dataclass(slots=True)
class Check:
    name: str
    status: Status
    detail: str
    hint: str | None = None
    data: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
            "hint": self.hint,
            **({"data": self.data} if self.data else {}),
        }


@dataclass(slots=True)
class Report:
    checks: list[Check] = field(default_factory=list)

    def add(self, check: Check) -> None:
        self.checks.append(check)

    @property
    def failed(self) -> list[Check]:
        return [check for check in self.checks if check.status == "fail"]

    @property
    def warnings(self) -> list[Check]:
        return [check for check in self.checks if check.status == "warn"]

    @property
    def exit_code(self) -> int:
        return 1 if self.failed else 0

    def describe(self) -> dict[str, Any]:
        return {
            "checks": [check.describe() for check in self.checks],
            "failed": len(self.failed),
            "warnings": len(self.warnings),
        }


def _port_open(host: str, port: int, timeout: float = 0.4) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, port)) == 0


async def run_diagnostics(config: Config | None = None) -> Report:
    """Run every check and collect the results."""
    config = config or load_config()
    report = Report()

    _check_config(config, report)
    _check_registry(config, report)
    await _check_upstream(config, report)
    await _check_dashboard(config, report)
    await _check_switchboard(config, report)
    _check_providers(config, report)
    _check_database(config, report)
    _check_privacy(config, report)
    _check_integrations(config, report)
    await _check_host(report)
    return report


def _check_config(config: Config, report: Report) -> None:
    if not config.config_path.is_file():
        report.add(
            Check(
                "config",
                "warn",
                f"{config.config_path} not found; built-in defaults are in use",
                hint="Copy config/switchboard.toml from the repository.",
            )
        )
    else:
        report.add(
            Check(
                "config",
                "ok",
                f"{config.config_path.name}: mode={config.routing.mode}, "
                f"port={config.server.port}, backends={', '.join(config.decisions.backends)}",
                data={
                    "mode": config.routing.mode,
                    "port": config.server.port,
                    "upstream": config.upstream.base_url,
                },
            )
        )

    unknown = [name for name in config.decisions.backends if name not in backend_names()]
    if unknown:
        report.add(
            Check(
                "config.backends",
                "fail",
                f"unknown decision backend(s): {', '.join(unknown)}",
                hint=f"Valid names: {', '.join(backend_names())}",
            )
        )
    if config.decisions.backends and config.decisions.backends[-1] != "rules":
        report.add(
            Check(
                "config.backends",
                "warn",
                "the last backend is not 'rules'",
                hint="Keep 'rules' last: it cannot fail, so it guarantees a decision.",
            )
        )


def _check_registry(config: Config, report: Report) -> None:
    snapshot = build_registry(config)
    usable = snapshot.routable()
    if not usable:
        report.add(
            Check(
                "models",
                "fail",
                f"no usable models found in {config.stack.models_dir}",
                hint="Check that the GGUF files exist and that models.json lists them.",
            )
        )
        return
    report.add(
        Check(
            "models",
            "ok",
            ", ".join(
                f"{model.id} ({model.size_gb:.1f} GB, ctx {model.context_limit}, "
                f"{'tools' if model.supports_tools else 'no tools'})"
                for model in usable
            ),
            data={"count": len(usable), "active_in_config": snapshot.active_model_file},
        )
    )
    for problem in snapshot.problems:
        report.add(
            Check(
                "models.problem",
                "warn",
                problem,
                hint="Stale entries in C:\\AI\\Config\\models.json are ignored, but the "
                "dashboard will still list them.",
            )
        )
    unmeasured = [model.id for model in usable if model.gen_tps is None]
    if unmeasured:
        report.add(
            Check(
                "models.measurements",
                "warn",
                f"no generation speed recorded for: {', '.join(unmeasured)}",
                hint="Run `switchboard benchmark` or let real traffic measure them.",
            )
        )


async def _check_upstream(config: Config, report: Report) -> None:
    from switchboard.upstream import LlamaClient

    client = LlamaClient(config)
    try:
        status = await client.status(force=True)
    finally:
        await client.aclose()

    if not status.healthy:
        report.add(
            Check(
                "llama-server",
                "fail",
                f"{config.upstream.base_url} is not responding ({status.error})",
                hint="Start the stack with C:\\AI\\start.ps1, then re-run doctor.",
            )
        )
        return

    detail = f"healthy at {config.upstream.base_url}, model={status.model_id}"
    if status.busy:
        detail += ", currently generating"
    if status.router_mode:
        detail += ", running in router mode"
    report.add(Check("llama-server", "ok", detail, data=status.describe()))

    if status.busy is None:
        report.add(
            Check(
                "llama-server.slots",
                "warn",
                "/slots is not available, so 'is it busy?' cannot be answered",
                hint="Without it Switchboard may switch models while another client "
                "is generating. Start llama-server with --slots (it is on by default).",
            )
        )


async def _check_dashboard(config: Config, report: Report) -> None:
    url = config.stack.dashboard_url.rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(4.0, connect=2.0)) as client:
            response = await client.get(f"{url}/api/models")
        ok = response.status_code == 200
    except httpx.HTTPError as error:
        report.add(
            Check(
                "dashboard",
                "warn" if config.routing.mode != "active" else "fail",
                f"{url} is not reachable ({type(error).__name__})",
                hint="The dashboard performs model switches. Without it Switchboard "
                "routes to whatever is already loaded. Start it with "
                "C:\\AI\\start.ps1 or dashboard-server.ps1.",
            )
        )
        return
    report.add(
        Check(
            "dashboard",
            "ok" if ok else "warn",
            f"{url} reachable (/api/models -> {response.status_code})",
        )
    )


async def _check_switchboard(config: Config, report: Report) -> None:
    host, port = config.server.host, config.server.port
    if not _port_open(host, port):
        report.add(
            Check(
                "switchboard",
                "warn",
                f"nothing is listening on {host}:{port}",
                hint="Start it with scripts\\start.ps1 (or `switchboard serve`).",
            )
        )
        return
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(5.0, connect=2.0)) as client:
            response = await client.get(f"{config.base_url}/health")
            payload = response.json()
    except (httpx.HTTPError, ValueError) as error:
        report.add(
            Check(
                "switchboard",
                "fail",
                f"something is listening on {host}:{port} but it is not Switchboard ({error})",
                hint="Another service may have taken the port; change server.port.",
            )
        )
        return
    report.add(
        Check(
            "switchboard",
            "ok",
            f"running at {config.base_url} (mode={payload.get('mode')}, "
            f"resident={payload.get('resident_model')})",
            data=payload,
        )
    )


def _check_providers(config: Config, report: Report) -> None:
    if config.jev_enabled():
        report.add(
            Check(
                "provider.jev",
                "ok",
                f"configured (model={config.decisions.jev.model}, "
                f"timeout={config.decisions.jev.timeout_s}s)",
            )
        )
    elif not config.typesafe_api_key:
        report.add(
            Check(
                "provider.jev",
                "warn",
                "unconfigured: TYPESAFE_API_KEY is not set",
                hint="Set it in C:\\AI\\Switchboard\\.env to enable cloud decisions. "
                "Switchboard works without it.",
            )
        )
    else:
        report.add(
            Check(
                "provider.jev",
                "warn",
                "disabled by configuration"
                + ("" if config.privacy.allow_cloud else " (privacy.allow_cloud is false)"),
            )
        )

    local = config.decisions.local
    report.add(
        Check(
            "provider.local",
            "ok" if local.enabled and "local" in config.decisions.backends else "warn",
            f"enabled={local.enabled}, skips when the resident model is slower than "
            f"{local.min_resident_gen_tps} tok/s"
            + (", skips while llama-server is busy" if local.skip_when_busy else ""),
        )
    )
    report.add(
        Check(
            "provider.logit",
            "skip" if not config.decisions.logit.enabled else "ok",
            "experimental backend is disabled"
            if not config.decisions.logit.enabled
            else "experimental backend is enabled",
            hint=None
            if not config.decisions.logit.enabled
            else "Validate it with `switchboard eval --backends logit` before trusting it.",
        )
    )
    report.add(
        Check(
            "provider.rules",
            "ok" if "rules" in config.decisions.backends else "fail",
            "deterministic fallback is in the chain"
            if "rules" in config.decisions.backends
            else "the deterministic fallback is NOT in the chain",
            hint=None
            if "rules" in config.decisions.backends
            else "Add 'rules' to decisions.backends so a decision is always possible.",
        )
    )


def _check_database(config: Config, report: Report) -> None:
    path = config.db_file
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path)
        try:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            tables = [
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
                )
            ]
            routes = 0
            if "routes" in tables:
                routes = connection.execute("SELECT COUNT(*) FROM routes").fetchone()[0]
        finally:
            connection.close()
    except (OSError, sqlite3.Error) as error:
        report.add(
            Check(
                "database",
                "fail",
                f"{path} is not usable: {error}",
                hint="Switchboard still routes without it, but records nothing.",
            )
        )
        return
    size_kb = path.stat().st_size / 1024 if path.exists() else 0
    report.add(
        Check(
            "database",
            "ok",
            f"{path} (schema v{version}, {len(tables)} tables, {routes} routes, {size_kb:.0f} KB)",
            data={"tables": tables, "routes": routes},
        )
    )


def _check_privacy(config: Config, report: Report) -> None:
    privacy = config.privacy
    cloud = "on" if (privacy.allow_cloud and config.jev_enabled()) else "off"
    report.add(
        Check(
            "privacy",
            "ok" if privacy.redact else "warn",
            f"redaction={'on' if privacy.redact else 'OFF'}, cloud decisions={cloud}, "
            f"state cap={privacy.max_state_chars} chars, "
            f"previews={'stored' if privacy.store_previews else 'not stored'} "
            f"({privacy.preview_chars} chars)",
            hint=None
            if privacy.redact
            else "privacy.redact is off: request text would leave this machine unredacted.",
        )
    )


def _check_integrations(config: Config, report: Report) -> None:
    # cptr: a connection pointing at Switchboard, and the MCP tool server.
    if not CPTR_CONFIG.is_file():
        report.add(Check("integration.cptr", "skip", f"{CPTR_CONFIG} not found"))
    else:
        try:
            text = CPTR_CONFIG.read_text(encoding="utf-8", errors="replace")
        except OSError as error:
            report.add(Check("integration.cptr", "warn", f"unreadable: {error}"))
        else:
            base = config.base_url
            has_connection = base in text
            has_mcp = f"{base}/mcp" in text
            report.add(
                Check(
                    "integration.cptr",
                    "ok" if has_connection else "warn",
                    f"connection={'present' if has_connection else 'absent'}, "
                    f"mcp tool server={'present' if has_mcp else 'absent'}",
                    hint=None
                    if has_connection
                    else "Run scripts\\integrate-cptr.ps1 (with cptr stopped) to add it.",
                )
            )

    start_script = config.stack.ai_root / "start.ps1"
    if start_script.is_file():
        try:
            content = start_script.read_text(encoding="utf-8", errors="replace")
        except OSError:
            content = ""
        present = "Switchboard" in content
        report.add(
            Check(
                "integration.start.ps1",
                "ok" if present else "warn",
                "start.ps1 starts Switchboard" if present else "start.ps1 does not start Switchboard",
                hint=None if present else "Run scripts\\integrate-start.ps1 to add an optional step.",
            )
        )


async def _check_host(report: Report) -> None:
    facts = await HostFacts().describe()
    ram_free = facts.get("ram_free_gb")
    vram_free = facts.get("vram_free_mb")
    detail = f"RAM {ram_free} GB free of {facts.get('ram_total_gb')} GB"
    if vram_free is not None:
        detail += f", VRAM {vram_free} MB free of {facts.get('vram_total_mb')} MB ({facts.get('gpu_name')})"
    status: Status = "ok"
    hint = None
    if isinstance(ram_free, (int, float)) and ram_free < 2.0:
        status = "warn"
        hint = (
            "Low free RAM. Partially offloaded models read their CPU layers from RAM on "
            "every token; this machine's tuning log recorded 9.6 -> 1.25 tok/s when RAM ran out."
        )
    report.add(Check("host", status, detail, hint=hint, data=facts))


def run_diagnostics_sync(config: Config | None = None) -> Report:
    return asyncio.run(run_diagnostics(config))


def format_report(report: Report, *, use_colour: bool = True) -> str:
    """Render the report for a terminal."""
    symbols = {"ok": "PASS", "warn": "WARN", "fail": "FAIL", "skip": "SKIP"}
    colours = {"ok": "\033[32m", "warn": "\033[33m", "fail": "\033[31m", "skip": "\033[90m"}
    reset = "\033[0m"
    lines: list[str] = []
    for check in report.checks:
        marker = symbols[check.status]
        if use_colour:
            marker = f"{colours[check.status]}{marker}{reset}"
        lines.append(f"[{marker}] {check.name:<26} {check.detail}")
        if check.hint:
            lines.append(f"       -> {check.hint}")
    lines.append("")
    lines.append(
        f"{len(report.checks)} checks, {len(report.failed)} failed, {len(report.warnings)} warnings"
    )
    return "\n".join(lines)


def report_json(report: Report) -> str:
    return json.dumps(report.describe(), indent=2)
