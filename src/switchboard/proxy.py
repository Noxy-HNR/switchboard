"""The OpenAI-compatible front door.

Anything that already speaks OpenAI - cptr, a script, an SDK - can point at
Switchboard instead of llama-server and get routing for free. The contract with
those clients is what shapes this module:

* **The body is forwarded, not rebuilt.** Only the ``model`` field is rewritten.
  Tool definitions, response formats, sampling settings and every field this
  code has never heard of reach llama.cpp untouched.
* **Streaming streams.** Upstream bytes are relayed as they arrive; nothing is
  buffered to inspect it. Usage is recovered by watching the bytes go past.
* **A model name is an instruction.** ``switchboard-auto`` (and the other
  virtual names) route; anything else is forwarded to that model.
* **Both path shapes work.** cptr appends ``/chat/completions`` directly to its
  base URL, so the endpoints exist with and without the ``/v1`` prefix.

The one place Switchboard inserts itself into the byte stream is while a model
loads: a streaming request emits SSE comment lines, which every compliant
parser ignores, so a 25-second load looks like a slow start rather than a dead
connection.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, AsyncIterator, Iterable

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from switchboard.decisions.base import TaskContext
from switchboard.policy import VIRTUAL_MODELS
from switchboard.privacy import hash_text
from switchboard.router import RoutedPlan, Router
from switchboard.upstream import UpstreamError

logger = logging.getLogger("switchboard.proxy")

# How long routing may take before a streaming client starts receiving
# keepalives instead of waiting on a silent socket.
_QUIET_WAIT_S = 2.0
_KEEPALIVE_EVERY_S = 5.0


def _text_of(content: Any) -> str:
    """Flatten OpenAI message content (string or content-part list) to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text" and isinstance(part.get("text"), str):
                    parts.append(part["text"])
                elif isinstance(part.get("text"), str):
                    parts.append(part["text"])
        return "\n".join(parts)
    return ""


def _has_images(messages: Iterable[dict[str, Any]]) -> bool:
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") in {"image_url", "image"}:
                    return True
    return False


def build_task_context(
    body: dict[str, Any], *, chars_per_token: float, session_id: str | None
) -> TaskContext:
    """Extract the routing inputs from an OpenAI chat request."""
    messages = [m for m in body.get("messages", []) if isinstance(m, dict)]
    user_messages = [m for m in messages if m.get("role") == "user"]
    system_messages = [m for m in messages if m.get("role") == "system"]

    latest = _text_of(user_messages[-1].get("content")) if user_messages else ""
    prior = _text_of(user_messages[-2].get("content")) if len(user_messages) > 1 else None
    system = _text_of(system_messages[0].get("content")) if system_messages else None

    tool_names: list[str] = []
    for tool in body.get("tools") or []:
        if isinstance(tool, dict):
            function = tool.get("function")
            if isinstance(function, dict) and isinstance(function.get("name"), str):
                tool_names.append(function["name"])
            elif isinstance(tool.get("name"), str):
                tool_names.append(tool["name"])

    # The whole conversation decides the context requirement, not just the last turn.
    total_chars = sum(len(_text_of(message.get("content"))) for message in messages)
    total_chars += sum(len(json.dumps(tool)) for tool in body.get("tools") or [])

    return TaskContext(
        message=latest,
        prior_message=prior,
        system_prompt=system,
        tool_names=tool_names,
        turn_index=len(user_messages),
        estimated_prompt_tokens=int(total_chars / max(1.0, chars_per_token)),
        has_images=_has_images(messages),
        session_id=session_id,
    )


def session_id_for(request: Request, body: dict[str, Any]) -> str | None:
    """Identify the conversation, without storing anything identifying.

    OpenAI requests carry no conversation id, so the fallback hashes the opening
    of the conversation: the system prompt plus the first user message stay
    constant as turns accumulate.
    """
    header = request.headers.get("x-switchboard-session")
    if header:
        return hash_text(header)
    user = body.get("user")
    if isinstance(user, str) and user:
        return hash_text(user)
    metadata = body.get("metadata")
    if isinstance(metadata, dict):
        for key in ("conversation_id", "session_id", "chat_id"):
            value = metadata.get(key)
            if isinstance(value, str) and value:
                return hash_text(value)

    messages = [m for m in body.get("messages", []) if isinstance(m, dict)]
    system = next((_text_of(m.get("content")) for m in messages if m.get("role") == "system"), "")
    first_user = next((_text_of(m.get("content")) for m in messages if m.get("role") == "user"), "")
    if not system and not first_user:
        return None
    return hash_text(f"{system[:2000]}␟{first_user[:2000]}")


class UsageWatcher:
    """Recovers token counts from a stream without buffering or altering it.

    llama.cpp only reports usage in a streamed response when the client asked
    for it, and rewriting the client's request to ask would change what the
    client receives. So: use the real usage when it happens to be there, and
    otherwise count content deltas, which llama.cpp emits one per token.
    """

    __slots__ = ("_buffer", "delta_chunks", "usage", "timings", "first_byte_at", "_started")

    def __init__(self) -> None:
        self._buffer = b""
        self.delta_chunks = 0
        self.usage: dict[str, Any] | None = None
        self.timings: dict[str, Any] | None = None
        self.first_byte_at: float | None = None
        self._started = time.perf_counter()

    def feed(self, chunk: bytes) -> None:
        if self.first_byte_at is None:
            self.first_byte_at = time.perf_counter()
        self._buffer += chunk
        while b"\n" in self._buffer:
            line, self._buffer = self._buffer.split(b"\n", 1)
            self._line(line.strip())
        # A line this long is not an SSE frame we can use. Drop the partial
        # line entirely and resync at the next newline: keeping a tail would
        # prepend garbage to the next frame and lose that one too.
        if len(self._buffer) > 1_000_000:
            self._buffer = b""

    def _line(self, line: bytes) -> None:
        if not line.startswith(b"data:"):
            return
        payload = line[5:].strip()
        if not payload or payload == b"[DONE]":
            return
        try:
            chunk = json.loads(payload)
        except (ValueError, UnicodeDecodeError):
            return
        if not isinstance(chunk, dict):
            return
        if isinstance(chunk.get("usage"), dict):
            self.usage = chunk["usage"]
        if isinstance(chunk.get("timings"), dict):
            self.timings = chunk["timings"]
        for choice in chunk.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta")
            if isinstance(delta, dict) and (
                delta.get("content") or delta.get("reasoning_content") or delta.get("tool_calls")
            ):
                self.delta_chunks += 1

    def result(self) -> tuple[int | None, int | None, float | None, float | None]:
        """(tokens_in, tokens_out, tokens/sec, ttfb_ms)."""
        elapsed = time.perf_counter() - (self.first_byte_at or self._started)
        tokens_in = tokens_out = None
        if self.usage:
            tokens_in = self.usage.get("prompt_tokens")
            tokens_out = self.usage.get("completion_tokens")
        if tokens_out is None and self.delta_chunks:
            tokens_out = self.delta_chunks
        tps = None
        if self.timings and isinstance(self.timings.get("predicted_per_second"), (int, float)):
            tps = float(self.timings["predicted_per_second"])
        elif tokens_out and elapsed > 0.5:
            tps = tokens_out / elapsed
        ttfb = (
            (self.first_byte_at - self._started) * 1000 if self.first_byte_at is not None else None
        )
        return tokens_in, tokens_out, tps, ttfb


def _route_headers(plan: RoutedPlan) -> dict[str, str]:
    """Routing metadata as headers, so the response body stays exactly upstream's."""

    def clean(value: str, limit: int = 400) -> str:
        # Header values must be latin-1 encodable and single-line.
        return " ".join(value.split())[:limit].encode("ascii", "replace").decode("ascii")

    headers = {
        "X-Switchboard-Model": plan.model_id or "unknown",
        "X-Switchboard-Backend": plan.decision.backend,
        "X-Switchboard-Reason": clean(plan.reason),
        "X-Switchboard-Domain": plan.decision.domain,
        "X-Switchboard-Difficulty": f"{plan.decision.difficulty:.2f}",
        "X-Switchboard-Switched": "true" if (plan.switch and plan.switch.ok) else "false",
    }
    if plan.route_id is not None:
        headers["X-Switchboard-Route-Id"] = str(plan.route_id)
    if plan.route.would_switch_to is not None:
        headers["X-Switchboard-Would-Switch-To"] = plan.route.would_switch_to.id
    return headers


def build_proxy_router(get_router) -> APIRouter:
    """OpenAI-compatible routes, registered under both '' and '/v1'."""
    api = APIRouter()

    @api.get("/models")
    async def list_models(request: Request) -> JSONResponse:
        router: Router = get_router(request)
        snapshot = router.registry()
        created = int(time.time())
        data: list[dict[str, Any]] = []
        for name, policy in VIRTUAL_MODELS.items():
            data.append(
                {
                    "id": name,
                    "object": "model",
                    "created": created,
                    "owned_by": "switchboard",
                    "switchboard": {"virtual": True, "policy": policy},
                }
            )
        for model in snapshot.models:
            if not model.exists:
                continue
            data.append(
                {
                    "id": model.id,
                    "object": "model",
                    "created": created,
                    "owned_by": "llama.cpp",
                    "switchboard": {
                        "virtual": False,
                        "role": model.role,
                        "context_limit": model.context_limit,
                        "supports_tools": model.supports_tools,
                        "file": model.filename,
                    },
                }
            )
        return JSONResponse({"object": "list", "data": data})

    @api.post("/chat/completions")
    async def chat_completions(request: Request):
        router: Router = get_router(request)
        try:
            body = await request.json()
        except ValueError:
            return JSONResponse(
                {"error": {"message": "request body is not valid JSON", "type": "invalid_request_error"}},
                status_code=400,
            )
        if not isinstance(body, dict) or not isinstance(body.get("messages"), list):
            return JSONResponse(
                {"error": {"message": "'messages' is required", "type": "invalid_request_error"}},
                status_code=400,
            )

        requested = body.get("model") if isinstance(body.get("model"), str) else None
        policy, explicit_model, recognised = router.resolve_requested_model(requested)
        streaming = bool(body.get("stream"))
        session_id = session_id_for(request, body)
        context = build_task_context(
            body,
            chars_per_token=router.config.policy.chars_per_token,
            session_id=session_id,
        )

        route_task = asyncio.ensure_future(
            router.route(
                context,
                policy=policy,
                explicit_model=explicit_model,
                requested_model=requested,
                max_tokens=body.get("max_tokens") or body.get("max_completion_tokens"),
                streaming=streaming,
                source="proxy",
            )
        )

        # Give routing a moment to finish quietly. It only takes longer when a
        # model is loading, and then a streaming client is better served by an
        # open stream with keepalives than by a silent wait.
        done, _ = await asyncio.wait({route_task}, timeout=_QUIET_WAIT_S)

        if streaming and not done:
            return StreamingResponse(
                _stream_with_keepalive(router, route_task, body, requested, recognised),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        plan: RoutedPlan = await route_task
        if not recognised and requested:
            plan.decision.notes.append(f"unknown model {requested!r}; routed automatically")

        if streaming:
            return StreamingResponse(
                _stream_response(router, plan, body),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", **_route_headers(plan)},
            )
        return await _complete(router, plan, body)

    return api


async def _prepare_body(router: Router, plan: RoutedPlan, body: dict[str, Any]) -> dict[str, Any]:
    """Copy the request with only the model name rewritten."""
    outgoing = dict(body)
    status, _ = await router.facts()
    # Prefer the id llama-server itself reports: in router mode it is what selects
    # the model, and on a single-model server it is simply echoed back.
    outgoing["model"] = status.model_id or (plan.model.filename if plan.model else body.get("model"))
    return outgoing


async def _complete(router: Router, plan: RoutedPlan, body: dict[str, Any]):
    """Non-streaming completion."""
    outgoing = await _prepare_body(router, plan, body)
    started = time.perf_counter()
    async with router.inflight():
        try:
            response = await router.upstream.chat_completion(outgoing)
        except UpstreamError as error:
            router.record_completion(plan.route_id, error=str(error), status_code=502)
            return JSONResponse(
                {
                    "error": {
                        "message": f"upstream llama-server error: {error}",
                        "type": "upstream_error",
                    }
                },
                status_code=502,
                headers=_route_headers(plan),
            )
    elapsed_ms = (time.perf_counter() - started) * 1000

    tokens_in = tokens_out = None
    gen_tps = None
    payload: Any = None
    try:
        payload = response.json()
    except ValueError:
        payload = None
    if isinstance(payload, dict):
        usage = payload.get("usage")
        if isinstance(usage, dict):
            tokens_in = usage.get("prompt_tokens")
            tokens_out = usage.get("completion_tokens")
        timings = payload.get("timings")
        if isinstance(timings, dict) and isinstance(
            timings.get("predicted_per_second"), (int, float)
        ):
            gen_tps = float(timings["predicted_per_second"])

    router.record_completion(
        plan.route_id,
        status_code=response.status_code,
        total_ms=elapsed_ms,
        ttfb_ms=elapsed_ms,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        gen_tps=gen_tps,
        model_id=plan.model_id,
    )

    headers = _route_headers(plan)
    if payload is None:
        return JSONResponse(
            {"error": {"message": "upstream returned a non-JSON response", "type": "upstream_error"}},
            status_code=502,
            headers=headers,
        )
    return JSONResponse(payload, status_code=response.status_code, headers=headers)


async def _stream_response(
    router: Router, plan: RoutedPlan, body: dict[str, Any]
) -> AsyncIterator[bytes]:
    """Relay upstream's stream byte for byte."""
    outgoing = await _prepare_body(router, plan, body)
    watcher = UsageWatcher()
    started = time.perf_counter()
    status_code = 200
    error: str | None = None

    async with router.inflight():
        try:
            async with router.upstream.stream_chat_completion(outgoing) as response:
                status_code = response.status_code
                if response.status_code >= 400:
                    detail = (await response.aread()).decode("utf-8", "replace")[:500]
                    error = f"upstream {response.status_code}: {detail}"
                    yield _sse_error(error)
                else:
                    async for chunk in response.aiter_raw():
                        watcher.feed(chunk)
                        yield chunk
        except (httpx.HTTPError, UpstreamError) as exception:
            error = f"{type(exception).__name__}: {exception}"
            logger.warning("stream failed: %s", error)
            yield _sse_error(error)

    tokens_in, tokens_out, tps, ttfb = watcher.result()
    router.record_completion(
        plan.route_id,
        status_code=status_code,
        ttfb_ms=ttfb,
        total_ms=(time.perf_counter() - started) * 1000,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        gen_tps=tps,
        error=error,
        model_id=plan.model_id,
    )


async def _stream_with_keepalive(
    router: Router,
    route_task: "asyncio.Future[RoutedPlan]",
    body: dict[str, Any],
    requested: str | None,
    recognised: bool,
) -> AsyncIterator[bytes]:
    """Hold the stream open with SSE comments while a model loads."""
    waited = 0.0
    while not route_task.done():
        done, _ = await asyncio.wait({route_task}, timeout=_KEEPALIVE_EVERY_S)
        if done:
            break
        waited += _KEEPALIVE_EVERY_S
        # A comment line: ignored by every compliant SSE parser, including the
        # one in cptr, which skips anything not starting with "data:".
        yield f": switchboard preparing a model ({waited:.0f}s)\n\n".encode()

    try:
        plan = await route_task
    except Exception as error:  # noqa: BLE001
        logger.exception("routing failed")
        yield _sse_error(f"routing failed: {type(error).__name__}: {error}")
        return

    if not recognised and requested:
        plan.decision.notes.append(f"unknown model {requested!r}; routed automatically")

    async for chunk in _stream_response(router, plan, body):
        yield chunk


def _sse_error(message: str) -> bytes:
    payload = json.dumps({"error": {"message": message, "type": "upstream_error"}})
    return f"data: {payload}\n\ndata: [DONE]\n\n".encode()
