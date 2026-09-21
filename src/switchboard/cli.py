"""The `switchboard` command.

Two kinds of command live here and they behave differently on purpose:

* **Commands about the machine** - ``doctor``, ``config-check``, ``models`` -
  work with Switchboard stopped. They are for the case where something is
  wrong, which is exactly when the service may not be running.
* **Commands about live behaviour** - ``status``, ``route``, ``step``,
  ``providers`` - talk to the running service over HTTP, so they observe the
  same router, cache and session state a real client would.

``route`` never switches models unless asked with ``--apply``: printing what
would happen has to be safe to run at any time.
"""

from __future__ import annotations

import asyncio
import json as json_module
import sys
from typing import Any, Optional

import httpx
import typer

from switchboard.config import load_config
from switchboard.diagnostics import format_report, report_json, run_diagnostics

app = typer.Typer(
    name="switchboard",
    help="Routing and supervisory layer for the local llama.cpp stack.",
    no_args_is_help=True,
    add_completion=False,
)

_JSON_OPTION = typer.Option(False, "--json", help="Print raw JSON instead of a summary.")


def _echo_json(payload: Any) -> None:
    typer.echo(json_module.dumps(payload, indent=2, default=str))


def _get(path: str, params: dict[str, Any] | None = None, timeout: float = 30.0) -> Any:
    config = load_config()
    url = f"{config.base_url}{path}"
    try:
        response = httpx.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as error:
        typer.secho(f"Switchboard is not answering at {url}: {error}", fg=typer.colors.RED)
        typer.secho("Start it with scripts\\start.ps1, or run `switchboard doctor`.", fg=typer.colors.YELLOW)
        raise typer.Exit(2) from error


def _post(path: str, payload: dict[str, Any], timeout: float = 600.0) -> Any:
    config = load_config()
    url = f"{config.base_url}{path}"
    try:
        response = httpx.post(url, json=payload, timeout=timeout)
        if response.status_code >= 400:
            typer.secho(f"{response.status_code}: {response.text[:400]}", fg=typer.colors.RED)
            raise typer.Exit(1)
        return response.json()
    except httpx.HTTPError as error:
        typer.secho(f"Switchboard is not answering at {url}: {error}", fg=typer.colors.RED)
        raise typer.Exit(2) from error


@app.command()
def serve(
    host: Optional[str] = typer.Option(None, help="Override server.host."),
    port: Optional[int] = typer.Option(None, help="Override server.port."),
    reload: bool = typer.Option(False, help="Reload on code changes (development)."),
) -> None:
    """Run the Switchboard service in the foreground."""
    import logging

    import uvicorn

    from switchboard.app import create_app

    config = load_config()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s %(message)s")
    uvicorn.run(
        "switchboard.app:create_app" if reload else create_app(config),
        factory=reload,
        host=host or config.server.host,
        port=port or config.server.port,
        reload=reload,
        access_log=False,
    )


@app.command()
def doctor(json: bool = _JSON_OPTION) -> None:
    """Diagnose the stack: llama-server, dashboard, models, providers, database."""
    report = asyncio.run(run_diagnostics())
    if json:
        typer.echo(report_json(report))
    else:
        typer.echo(format_report(report, use_colour=sys.stdout.isatty()))
    raise typer.Exit(report.exit_code)


@app.command("config-check")
def config_check(json: bool = _JSON_OPTION) -> None:
    """Show the effective configuration (secrets are never printed)."""
    config = load_config()
    payload = config.model_dump(mode="json", exclude={"typesafe_api_key"})
    payload["typesafe_api_key_present"] = bool(config.typesafe_api_key)
    payload["jev_enabled"] = config.jev_enabled()
    if json:
        _echo_json(payload)
        return
    typer.echo(f"config file      {config.config_path}")
    typer.echo(f"models overlay   {config.models_path}")
    typer.echo(f"listen           {config.base_url}")
    typer.echo(f"upstream         {config.upstream.base_url}")
    typer.echo(f"dashboard        {config.stack.dashboard_url}")
    typer.echo(f"routing mode     {config.routing.mode}")
    typer.echo(f"decision chain   {' -> '.join(config.decisions.backends)}")
    typer.echo(f"budget           {config.decisions.budget_ms} ms")
    typer.echo(f"jev              {'enabled' if config.jev_enabled() else 'not enabled'}"
               f" (TYPESAFE_API_KEY {'set' if config.typesafe_api_key else 'not set'})")
    typer.echo(f"privacy          redact={config.privacy.redact} cloud={config.privacy.allow_cloud} "
               f"cap={config.privacy.max_state_chars} previews={config.privacy.store_previews}")
    typer.echo(f"database         {config.db_file}")


@app.command()
def status(json: bool = _JSON_OPTION) -> None:
    """Show what is loaded, what is healthy, and which providers answer."""
    payload = _get("/status")
    if json:
        _echo_json(payload)
        return
    upstream = payload.get("upstream", {})
    typer.echo(f"mode            {payload.get('mode')}")
    typer.echo(f"resident model  {payload.get('resident_model')}")
    typer.echo(
        f"llama-server    {'healthy' if upstream.get('healthy') else 'DOWN'}"
        f"  busy={upstream.get('busy')}  {upstream.get('model_id') or ''}"
    )
    executor = payload.get("executor", {})
    typer.echo(f"switch executor {executor.get('name')}: {executor.get('detail')}")
    typer.echo(f"in flight       {payload.get('inflight')}")
    typer.echo("providers")
    for name, detail in (payload.get("providers") or {}).items():
        typer.echo(f"  {name:<8} {detail.get('availability'):<12} {detail.get('reason') or ''}")
    host = payload.get("host", {})
    typer.echo(
        f"host            RAM {host.get('ram_free_gb')}/{host.get('ram_total_gb')} GB free, "
        f"VRAM {host.get('vram_free_mb')}/{host.get('vram_total_mb')} MB free"
    )
    for problem in payload.get("registry_problems") or []:
        typer.secho(f"  ! {problem}", fg=typer.colors.YELLOW)


@app.command()
def models(
    refresh: bool = typer.Option(False, help="Re-read model files and configuration."),
    json: bool = _JSON_OPTION,
) -> None:
    """List the models Switchboard can route to."""
    try:
        payload = _get("/registry", {"refresh": refresh})
    except typer.Exit:
        # Works offline too: the registry does not need the service.
        from switchboard.registry import build_registry

        config = load_config()
        snapshot = build_registry(config)
        payload = {
            "resident": None,
            "active_in_stack_config": snapshot.active_model_file,
            "models": [model.describe() for model in snapshot.models],
            "problems": snapshot.problems,
        }
        typer.secho("(service not running: showing the registry read from disk)", fg=typer.colors.YELLOW)

    if json:
        _echo_json(payload)
        return
    resident = payload.get("resident")
    typer.echo(f"{'model':<18}{'role':<18}{'ctx':>9}{'size':>8}  tools  speed        state")
    for model in payload.get("models", []):
        speed = model.get("gen_tps") or {}
        speed_text = f"{speed.get('value')} tok/s" if speed else "unmeasured"
        state = []
        if model["id"] == resident:
            state.append("RESIDENT")
        if not model["exists"]:
            state.append("MISSING")
        elif not model["enabled"]:
            state.append("disabled")
        typer.echo(
            f"{model['id']:<18}{model['role']:<18}{str(model['context_limit']):>9}"
            f"{model['size_gb']:>7.1f}G  {'yes' if model['supports_tools'] else ' no':<5}"
            f"  {speed_text:<12} {' '.join(state)}"
        )
    for problem in payload.get("problems") or []:
        typer.secho(f"  ! {problem}", fg=typer.colors.YELLOW)


@app.command()
def providers(
    probe: bool = typer.Option(True, help="Re-probe availability instead of using cached state."),
    json: bool = _JSON_OPTION,
) -> None:
    """Show decision backend availability, breaker state and recent cost."""
    payload = _get("/providers", {"probe": probe})
    if json:
        _echo_json(payload)
        return
    typer.echo(f"chain order: {' -> '.join(payload.get('order', []))}")
    for name, detail in (payload.get("health") or {}).items():
        typer.echo(
            f"  {name:<8} {detail.get('availability'):<12} "
            f"failures={detail.get('consecutive_failures')} "
            f"{'BREAKER OPEN' if detail.get('open') else ''} {detail.get('reason') or ''}"
        )
    stats = payload.get("last_24h") or {}
    if stats:
        typer.echo("last 24h:")
        for name, detail in stats.items():
            typer.echo(
                f"  {name:<8} attempts={detail.get('attempts')} ok={detail.get('ok')} "
                f"p50={detail.get('p50_latency_ms')}ms p95={detail.get('p95_latency_ms')}ms "
                f"cost=${detail.get('est_cost_usd') or 0:.4f}"
            )


@app.command()
def route(
    message: str = typer.Argument(..., help="The request to route."),
    policy: str = typer.Option("auto", help="auto, fast, deep or medical."),
    tools: Optional[str] = typer.Option(None, help="Comma-separated tool names available."),
    session: Optional[str] = typer.Option(None, help="Conversation id, for stickiness."),
    apply: bool = typer.Option(False, "--apply", help="Actually switch models if warranted."),
    json: bool = _JSON_OPTION,
) -> None:
    """Ask which model should handle a request."""
    payload = _post(
        "/agent/route",
        {
            "message": message,
            "policy": policy,
            "tools": [name.strip() for name in (tools or "").split(",") if name.strip()],
            "session": session,
            "apply": apply,
        },
    )
    if json:
        _echo_json(payload)
        return
    decision = payload.get("decision", {})
    typer.secho(f"model      {payload.get('selected_model')}", fg=typer.colors.GREEN)
    typer.echo(f"reason     {payload.get('reason')}")
    typer.echo(
        f"judgment   domain={decision.get('domain')} "
        f"difficulty={decision.get('difficulty')} "
        f"tools={decision.get('tool_need')} ambiguity={decision.get('ambiguity')} "
        f"(by {decision.get('backend')}, {decision.get('latency_ms')} ms)"
    )
    routing = payload.get("routing", {})
    if routing.get("would_switch_to"):
        typer.secho(
            f"would switch to {routing['would_switch_to']} (blocked by {routing.get('blocked_by')})",
            fg=typer.colors.YELLOW,
        )
    switch = payload.get("switch")
    if switch:
        typer.echo(f"switch     {switch}")


@app.command()
def step(
    goal: str = typer.Option(..., help="What the overall task is trying to achieve."),
    action: str = typer.Option(..., help="What this step did."),
    result: str = typer.Option(..., help="What came back."),
    exit_code: Optional[int] = typer.Option(None, help="Exit code, if a command ran."),
    attempt: int = typer.Option(1, help="Which attempt this was."),
    json: bool = _JSON_OPTION,
) -> None:
    """Evaluate an agent step and get a recommendation."""
    payload = _post(
        "/agent/step",
        {
            "goal": goal,
            "action": action,
            "result": result,
            "exit_code": exit_code,
            "attempt": attempt,
        },
    )
    if json:
        _echo_json(payload)
        return
    decision = payload.get("decision", {})
    typer.secho(f"recommendation  {payload.get('recommendation')}", fg=typer.colors.GREEN)
    typer.echo(f"rationale       {payload.get('rationale')}")
    typer.echo(
        f"judgment        outcome={decision.get('outcome')} progress={decision.get('progress')} "
        f"complete={decision.get('complete')} (by {decision.get('backend')})"
    )
    if payload.get("escalate_to"):
        typer.echo(f"escalate to     {payload['escalate_to']}")


@app.command()
def switch(
    model: str = typer.Argument(..., help="Model id, alias or GGUF filename."),
    json: bool = _JSON_OPTION,
) -> None:
    """Load a specific model (restarts llama-server through the dashboard)."""
    payload = _post("/agent/switch", {"model": model})
    if json:
        _echo_json(payload)
        return
    typer.secho(
        f"loaded {payload.get('to_model')} in {(payload.get('load_ms') or 0) / 1000:.1f}s",
        fg=typer.colors.GREEN,
    )


@app.command()
def recent(limit: int = typer.Option(15), json: bool = _JSON_OPTION) -> None:
    """Show recent routing decisions."""
    payload = _get("/routes", {"limit": limit})
    routes = payload.get("routes", [])
    if json:
        _echo_json(routes)
        return
    typer.echo(f"{'when':<20}{'model':<16}{'backend':<8}{'ms':>6}  reason")
    for row in routes:
        when = str(row.get("ts", ""))[:19]
        typer.echo(
            f"{when:<20}{str(row.get('selected_model')):<16}{str(row.get('decision_backend')):<8}"
            f"{row.get('decision_latency_ms') or 0:>6.0f}  {str(row.get('reason'))[:70]}"
        )


@app.command()
def stats(hours: float = typer.Option(24.0), json: bool = _JSON_OPTION) -> None:
    """Aggregate routing and provider statistics."""
    payload = _get("/stats", {"hours": hours})
    if json:
        _echo_json(payload)
        return
    _echo_json(payload)


@app.command("eval")
def run_eval(
    backends: Optional[str] = typer.Option(
        None, help="Comma-separated backends to compare, e.g. rules,jev. Default: the configured chain."
    ),
    fixtures: Optional[str] = typer.Option(None, help="Path to a fixtures file."),
    out: Optional[str] = typer.Option(None, help="Directory for the report files."),
    limit: Optional[int] = typer.Option(None, help="Only run the first N fixtures."),
) -> None:
    """Run the routing eval suite and write a report."""
    from switchboard.evals.runner import run_eval_cli

    raise typer.Exit(
        run_eval_cli(
            backends=[name.strip() for name in backends.split(",")] if backends else None,
            fixtures=fixtures,
            out=out,
            limit=limit,
        )
    )


@app.command()
def benchmark(
    models: Optional[str] = typer.Option(None, help="Comma-separated model ids. Default: resident only."),
    switches: bool = typer.Option(
        False, "--switches", help="Measure real model load times (restarts llama-server)."
    ),
    tokens: int = typer.Option(64, help="Tokens to generate per speed sample."),
    out: Optional[str] = typer.Option(None, help="Directory for the report files."),
) -> None:
    """Measure generation speed, and optionally model load times."""
    from switchboard.evals.benchmark import run_benchmark_cli

    raise typer.Exit(
        run_benchmark_cli(
            models=[name.strip() for name in models.split(",")] if models else None,
            measure_switches=switches,
            tokens=tokens,
            out=out,
        )
    )


@app.command()
def prune(days: Optional[int] = typer.Option(None, help="Override the retention window.")) -> None:
    """Delete observability rows older than the retention window."""
    from switchboard.store import Store

    config = load_config()
    store = Store(config.db_file, preview_chars=config.privacy.preview_chars)
    try:
        deleted = store.prune(days if days is not None else config.observability.retention_days)
    finally:
        store.close()
    _echo_json(deleted)


def main() -> None:
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
