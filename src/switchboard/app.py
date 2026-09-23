"""The FastAPI application.

One process serves four surfaces, because they all need the same router and a
second process would mean a second view of which model is resident:

* ``/v1/...`` and ``/...``  - the OpenAI-compatible proxy (cptr omits ``/v1``);
* ``/status``, ``/agent/...`` - the supervisory API;
* ``/mcp``                   - the MCP server, for agents that speak it;
* ``/dashboard``             - a single page for looking at what happened.

Start-up order matters: the store opens first (the registry reads measurements
from it), then the upstream client, then the decision chain, then the router.
Nothing here contacts llama-server at import time, so Switchboard starts
whether or not the stack is running.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

from switchboard.api import build_api_router
from switchboard.config import Config, PROJECT_ROOT, load_config
from switchboard.decisions.factory import build_chain
from switchboard.executor import build_executor
from switchboard.proxy import build_proxy_router
from switchboard.router import Router
from switchboard.security import LocalBoundary
from switchboard.settings import build_settings_router
from switchboard.store import Store
from switchboard.upstream import LlamaClient

logger = logging.getLogger("switchboard")

DASHBOARD_FILE = Path(__file__).resolve().parent / "static" / "dashboard.html"


def get_router(request: Request) -> Router:
    router = getattr(request.app.state, "router", None)
    if router is None:  # pragma: no cover - only reachable if startup failed
        raise RuntimeError("Switchboard router is not initialised")
    return router


def create_app(config: Config | None = None) -> FastAPI:
    """Build the application. Tests call this directly with their own Config."""
    config = config or load_config()
    mcp = _create_mcp()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        store: Store | None
        try:
            store = Store(config.db_file, preview_chars=config.privacy.preview_chars)
        except Exception as error:  # noqa: BLE001 - observability is not worth refusing to start
            logger.warning("could not open the store at %s: %s", config.db_file, error)
            store = None

        upstream = LlamaClient(config)
        chain = build_chain(config, upstream)
        executor = build_executor(config, upstream)
        router = Router(
            config, upstream=upstream, chain=chain, executor=executor, store=store
        )
        app.state.config = config
        app.state.router = router
        app.state.store = store
        _MCP_HOLDER["router"] = router

        # Build the registry once at start-up so the first request is not the
        # one that pays for reading three GGUF headers.
        router.registry(refresh=True)
        logger.info(
            "Switchboard listening on %s in %s mode; upstream %s",
            config.base_url,
            config.routing.mode,
            config.upstream.base_url,
        )
        try:
            if mcp is not None:
                # A mounted sub-application does not get its own lifespan, so the
                # MCP session manager has to be run from the parent's.
                async with mcp.session_manager.run():
                    yield
            else:
                yield
        finally:
            # The dashboard may have swapped the chain since start-up.
            await router.aclose_retired()
            await router.chain.aclose()
            close = getattr(executor, "aclose", None)
            if close is not None:
                await close()
            await upstream.aclose()
            if store is not None:
                store.close()

    app = FastAPI(
        title="Switchboard",
        version="0.1.0",
        summary="Routing and supervisory layer for the local llama.cpp stack",
        description=(
            "Switchboard decides which local model should answer a request, whether a "
            "switch is worth its cost, and whether an agent step made progress. It "
            "speaks the OpenAI API so existing clients can use it unchanged."
        ),
        lifespan=lifespan,
    )

    proxy = build_proxy_router(get_router)
    # cptr builds "{base_url}/chat/completions", other clients use "/v1/...".
    app.include_router(proxy, prefix="/v1", tags=["openai"])
    app.include_router(proxy, tags=["openai (no /v1 prefix)"])
    app.include_router(build_api_router(get_router), tags=["switchboard"])
    app.include_router(build_settings_router(get_router), tags=["settings"])

    @app.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
    async def dashboard() -> HTMLResponse:
        if not DASHBOARD_FILE.is_file():  # pragma: no cover
            return HTMLResponse("<h1>Dashboard not installed</h1>", status_code=404)
        # The C:\AI dashboard shows the settings panel in a frame; no other page
        # may frame it, so a site cannot overlay it and steer clicks on it.
        return HTMLResponse(
            DASHBOARD_FILE.read_text(encoding="utf-8"),
            headers={"Content-Security-Policy": frame_ancestors(config)},
        )

    @app.get("/", include_in_schema=False)
    async def index(request: Request) -> JSONResponse:
        # Settings saved from the dashboard replace app.state.config.
        config_now: Config = getattr(request.app.state, "config", None) or config
        return JSONResponse(
            {
                "service": "switchboard",
                "mode": config_now.routing.mode,
                "openai_base_url": f"{config.base_url}/v1",
                "virtual_models": ["switchboard-auto", "switchboard-fast", "switchboard-deep", "switchboard-medical"],
                "docs": f"{config.base_url}/docs",
                "dashboard": f"{config.base_url}/dashboard",
                "status": f"{config.base_url}/status",
            }
        )

    if mcp is not None:
        app.mount("/mcp", mcp.streamable_http_app())

    # Outermost, so it sees every request first - including /mcp. A web page the
    # user happens to have open must not be able to spend GPU time or trigger a
    # model restart through the browser, and no request body is allowed to grow
    # without bound. Native clients (cptr, SDKs, curl) send no Origin at all.
    app.add_middleware(LocalBoundary, port=config.server.port, host=config.server.host)
    return app


def frame_ancestors(config: Config) -> str:
    """Who may frame the dashboard: itself and the stack's dashboard (:9090), on either loopback name."""
    allowed = ["'self'"]
    parsed = urlsplit(config.stack.dashboard_url)
    if parsed.scheme in ("http", "https") and parsed.hostname:
        port = f":{parsed.port}" if parsed.port else ""
        hosts = [parsed.hostname]
        if parsed.hostname in ("127.0.0.1", "localhost"):
            hosts = ["127.0.0.1", "localhost"]
        allowed.extend(f"{parsed.scheme}://{host}{port}" for host in hosts)
    return "frame-ancestors " + " ".join(allowed)


_MCP_HOLDER: dict[str, Any] = {}


def _create_mcp():
    """Build the MCP server, tolerating an environment without the mcp package.

    The tools need the live router, which only exists once the app has started,
    so they resolve it through the holder rather than capturing it.
    """
    try:
        from switchboard.mcp_server import create_mcp
    except Exception as error:  # noqa: BLE001 - MCP is optional at runtime
        logger.warning("MCP server not available: %s", error)
        return None
    try:
        return create_mcp(lambda: _MCP_HOLDER["router"])
    except Exception as error:  # noqa: BLE001
        logger.warning("MCP server could not be built: %s", error)
        return None


def run() -> None:  # pragma: no cover - exercised by scripts/start.ps1
    """Entry point used by `switchboard serve`."""
    import uvicorn

    config = load_config()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s %(message)s",
    )
    uvicorn.run(
        create_app(config),
        host=config.server.host,
        port=config.server.port,
        log_level="info",
        access_log=False,
    )


__all__ = ["create_app", "run", "get_router", "PROJECT_ROOT"]
