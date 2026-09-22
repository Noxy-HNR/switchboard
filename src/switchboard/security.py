"""The local boundary: who may call Switchboard, and with how much.

Switchboard listens on 127.0.0.1 and has no authentication, which is right for
a single-user machine and would still leave one gap: any web page open in a
browser on this machine can send requests to localhost. A page doing that could
spend GPU time or, worse, trigger a model restart. Native clients - cptr, the
OpenAI SDKs, curl, MCP clients - send no ``Origin`` header at all, while every
browser does, so the rule is simple:

* a request that carries an ``Origin`` must come from Switchboard's own
  origin (the dashboard), or it is refused;
* a request with a body must declare ``application/json`` - which a browser
  cannot send cross-origin without a preflight - and stay under a size cap.

This is protection for a local service, not authentication for a networked
one. Exposing Switchboard beyond loopback would need real credentials.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable
from urllib.parse import urlsplit

from starlette.responses import JSONResponse

Scope = dict[str, Any]
Message = dict[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

DEFAULT_MAX_BODY_BYTES = 8 * 1024 * 1024
_BODY_METHODS = {"POST", "PUT", "PATCH"}
_LOOPBACK_NAMES = {"localhost", "127.0.0.1", "::1"}


def _error(message: str, status: int) -> JSONResponse:
    return JSONResponse({"error": {"message": message, "type": "request_refused"}}, status_code=status)


class LocalBoundary:
    """ASGI middleware enforcing the rules in the module docstring."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        port: int,
        host: str,
        max_bytes: int = DEFAULT_MAX_BODY_BYTES,
    ) -> None:
        self.app = app
        self.port = port
        self.host = host
        self.max_bytes = max_bytes

    def _origin_allowed(self, origin: bytes) -> bool:
        """Only Switchboard's own origin: http, loopback host, this port, nothing else."""
        try:
            parsed = urlsplit(origin.decode("ascii"))
            # .port raises ValueError for "localhost:8002.evil.example".
            port = parsed.port or 80
        except (ValueError, UnicodeError):
            return False
        return (
            parsed.scheme == "http"
            and parsed.hostname in {self.host, *_LOOPBACK_NAMES}
            and port == self.port
            and not parsed.username
            and not parsed.path
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        origin = headers.get(b"origin")
        if origin is not None and not self._origin_allowed(origin):
            await _error("Cross-origin request refused", 403)(scope, receive, send)
            return

        if scope["method"] not in _BODY_METHODS:
            await self.app(scope, receive, send)
            return

        content_type = headers.get(b"content-type", b"").split(b";", 1)[0].strip().lower()
        if content_type != b"application/json":
            await _error("application/json required", 415)(scope, receive, send)
            return

        # Read the (bounded) request body up front. Counting the bytes rather
        # than trusting Content-Length also covers chunked uploads. Only the
        # request is buffered; responses, streamed ones included, pass straight
        # through untouched.
        chunks: list[bytes] = []
        size = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            size += len(chunk)
            if size > self.max_bytes:
                await _error("Request body too large", 413)(scope, receive, send)
                return
            chunks.append(chunk)
            if not message.get("more_body", False):
                break

        delivered = False

        async def replay() -> Message:
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": b"".join(chunks), "more_body": False}
            # After the body, pass through so disconnects are still observed.
            return await receive()

        await self.app(scope, replay, send)
