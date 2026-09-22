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
  virtual names) route; any other name must be a model the registry knows, and
  is forwarded to that model. An unknown name gets the OpenAI API's own answer,
  a 404 ``model_not_found``, before any routing: quietly routing it would be a
  substitution. Only a missing or empty ``model`` field routes by default.
* **An explicit choice fails rather than substitutes.** When the named model
  cannot be loaded, the request is refused with a 409 ``model_unavailable``
  error. Answering on whatever happens to be resident would hand the client a
  model it did not ask for under a status that says everything went fine.
* **Residency is re-checked under the reservation.** Routing reads the upstream
  before the request is counted in flight, so a switch can still land in
  between. Once ``router.inflight()`` has registered the request no switch can
  begin, so the upstream is read once more there: a different resident model
  means the request is refused before anything is generated, and that same read
  supplies the outgoing ``model`` field.
* **Nothing llama-server sends back is persisted.** Its error bodies can quote
  the prompt, so the database records only the HTTP status or the exception's
  type name. The client still receives the upstream's own detail.
* **Every request records how it ended**, including a client that disconnects
  mid-stream, and routing that nobody is waiting for any more is cancelled.
* **Both path shapes work.** cptr appends ``/chat/completions`` directly to its
  base URL, so the endpoints exist with and without the ``/v1`` prefix.

The one place Switchboard inserts itself into the byte stream is while a model
loads: a streaming request emits SSE comment lines, which every compliant
parser ignores, so a 25-second load looks like a slow start rather than a dead
connection. Because that stream has already answered 200, a refusal found after
the load arrives as an SSE error frame followed by ``[DONE]`` instead.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections.abc import AsyncGenerator, AsyncIterator, Iterable
from dataclasses import dataclass
from typing import Any

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from switchboard.decisions.base import TaskContext
from switchboard.policy import VIRTUAL_MODELS
from switchboard.privacy import hash_text
from switchboard.registry import ModelEntry
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
        if requested and requested.strip() and not recognised:
            # A name the client chose is an instruction too. Routing it
            # automatically would be a substitution, so answer the way the
            # OpenAI API does - before any decision is made or recorded.
            return JSONResponse(
                {
                    "error": {
                        "message": f"The model '{requested}' does not exist",
                        "type": "invalid_request_error",
                        "code": "model_not_found",
                    }
                },
                status_code=404,
            )
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

        if streaming:
            refusal = _explicit_refusal(plan)
            if refusal is not None:
                # Nothing has been sent yet, so this can still be a real status.
                return _refuse(router, plan, refusal)
            return StreamingResponse(
                _stream_response(router, plan, body),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", **_route_headers(plan)},
            )
        return await _complete(router, plan, body)

    return api


@dataclass(frozen=True, slots=True)
class _Refusal:
    """A request Switchboard declines to forward, and how that is reported."""

    # What the client is told. It may name models; it never quotes the request.
    message: str
    # What the route row records.
    recorded: str
    # The OpenAI-style error type.
    kind: str


def _explicit_refusal(plan: RoutedPlan) -> _Refusal | None:
    """Refuse a request whose named model could not be loaded.

    The router falls back to the resident model when a switch fails or is not
    allowed, which is right for automatic routing and wrong for a request that
    named its model: forwarding it would answer with a model the client did not
    choose, and nothing in a 200 response would say so.
    """
    route = plan.route
    if not route.explicit:
        return None
    if route.would_switch_to is not None:
        wanted = route.would_switch_to.id
        cause = "Switchboard cannot switch models right now"
    elif plan.switch is not None and not plan.switch.ok:
        wanted = plan.switch.to_model
        cause = "the switch failed"
        if plan.switch.error:
            cause += f": {plan.switch.error}"
    else:
        return None
    return _Refusal(
        message=(
            f"{wanted} was requested but could not be loaded ({cause}); "
            "the request was not sent to another model."
        ),
        recorded=f"requested model {wanted} could not be loaded",
        kind="model_unavailable",
    )


def _residency_drift(plan: RoutedPlan, resident: ModelEntry | None) -> _Refusal | None:
    """Refuse a request whose routed model is no longer the one loaded.

    An unknown resident (llama-server down or restarting) is not drift: the
    forward then fails on its own with the ordinary unreachable error.
    """
    if plan.model is None or resident is None or resident.id == plan.model.id:
        return None
    return _Refusal(
        message=(
            f"This request was routed to {plan.model.id}, but llama-server now has "
            f"{resident.id} loaded, so it was not sent. Retrying routes it again."
        ),
        recorded=f"resident model changed to {resident.id} before generation",
        kind="upstream_error",
    )


def _error_body(message: str, kind: str) -> dict[str, Any]:
    return {"error": {"message": message, "type": kind}}


def _refuse(router: Router, plan: RoutedPlan, refusal: _Refusal) -> JSONResponse:
    router.record_completion(plan.route_id, status_code=409, error=refusal.recorded)
    headers = _route_headers(plan)
    # No model answered this request, so the header must not name one.
    headers["X-Switchboard-Model"] = "none"
    headers["X-Switchboard-Refused"] = refusal.kind
    return JSONResponse(_error_body(refusal.message, refusal.kind), status_code=409, headers=headers)


def _failure_kind(error: BaseException) -> str:
    """The exception's type name, which is all of it the database may keep.

    ``UpstreamError`` wraps the httpx exception that actually happened, and
    that one's name (``ConnectError``, ``ReadTimeout``) is the useful part.
    """
    if isinstance(error, UpstreamError) and error.__cause__ is not None:
        return type(error.__cause__).__name__
    return type(error).__name__


def _prepare_body(plan: RoutedPlan, body: dict[str, Any], resident_id: str | None) -> dict[str, Any]:
    """Copy the request with only the model name rewritten.

    ``resident_id`` must be the read taken under the in-flight reservation, so
    the name sent is the one just checked for drift, not an older cached one.
    """
    outgoing = dict(body)
    # Prefer the id llama-server itself reports: in router mode it is what selects
    # the model, and on a single-model server it is simply echoed back.
    outgoing["model"] = resident_id or (plan.model.filename if plan.model else body.get("model"))
    return outgoing


async def _complete(router: Router, plan: RoutedPlan, body: dict[str, Any]):
    """Non-streaming completion."""
    refusal = _explicit_refusal(plan)
    if refusal is not None:
        return _refuse(router, plan, refusal)

    headers = _route_headers(plan)
    started = time.perf_counter()
    async with router.inflight():
        model_id, resident = await router.resident_now()
        drift = _residency_drift(plan, resident)
        if drift is not None:
            return _refuse(router, plan, drift)
        outgoing = _prepare_body(plan, body, model_id)
        try:
            response = await router.upstream.chat_completion(outgoing)
        except UpstreamError as error:
            router.record_completion(
                plan.route_id,
                status_code=502,
                error=f"upstream unreachable ({_failure_kind(error)})",
            )
            return JSONResponse(
                _error_body(f"upstream llama-server error: {error}", "upstream_error"),
                status_code=502,
                headers=headers,
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

    failure: str | None = None
    if response.status_code >= 400:
        # The client gets llama-server's body verbatim below; the database gets
        # only the status, because that body can quote the prompt back.
        failure = f"upstream returned HTTP {response.status_code}"
    elif payload is None:
        failure = "upstream returned a non-JSON response"

    router.record_completion(
        plan.route_id,
        status_code=response.status_code,
        total_ms=elapsed_ms,
        ttfb_ms=elapsed_ms,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        gen_tps=gen_tps,
        error=failure,
        model_id=plan.model_id,
    )

    if payload is None:
        return JSONResponse(
            _error_body("upstream returned a non-JSON response", "upstream_error"),
            status_code=502,
            headers=headers,
        )
    return JSONResponse(payload, status_code=response.status_code, headers=headers)


async def _stream_response(
    router: Router, plan: RoutedPlan, body: dict[str, Any]
) -> AsyncGenerator[bytes, None]:
    """Relay upstream's stream byte for byte, and record how it ended."""
    watcher = UsageWatcher()
    started = time.perf_counter()
    status_code: int | None = None
    error: str | None = None

    try:
        # Only the keepalive path gets here with a refusal: it has already sent
        # 200. The handler turns every other streaming refusal into a real 409.
        refusal = _explicit_refusal(plan)
        if refusal is not None:
            status_code, error = 409, refusal.recorded
            yield _sse_error(refusal.message, kind=refusal.kind)
            return

        async with router.inflight():
            model_id, resident = await router.resident_now()
            drift = _residency_drift(plan, resident)
            if drift is not None:
                status_code, error = 409, drift.recorded
                yield _sse_error(drift.message, kind=drift.kind)
                return
            outgoing = _prepare_body(plan, body, model_id)
            try:
                async with router.upstream.stream_chat_completion(outgoing) as response:
                    status_code = response.status_code
                    if response.status_code >= 400:
                        detail = (await response.aread()).decode("utf-8", "replace")[:500]
                        # The client sees llama-server's words; the database does not.
                        error = f"upstream returned HTTP {response.status_code}"
                        yield _sse_error(f"upstream {response.status_code}: {detail}")
                    else:
                        async for chunk in response.aiter_raw():
                            watcher.feed(chunk)
                            yield chunk
            except (httpx.HTTPError, UpstreamError) as exception:
                if status_code is None:
                    status_code = 502
                    error = f"upstream unreachable ({_failure_kind(exception)})"
                else:
                    error = f"upstream stream failed ({_failure_kind(exception)})"
                logger.warning("stream failed: %s", error)
                yield _sse_error(f"{type(exception).__name__}: {exception}")
    except (GeneratorExit, asyncio.CancelledError):
        # The client went away at a yield, or its task was cancelled at an
        # await. Record what was seen, then let the exit carry on: swallowing it
        # would break aclose() or lose the task's cancellation.
        if error is None:
            error = "client disconnected"
        raise
    except Exception as exception:
        if error is None:
            error = f"proxy failed ({type(exception).__name__})"
        raise
    finally:
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
    handed_off = False
    try:
        waited = 0.0
        while not route_task.done():
            done, _ = await asyncio.wait({route_task}, timeout=_KEEPALIVE_EVERY_S)
            if done:
                break
            waited += _KEEPALIVE_EVERY_S
            # A comment line: ignored by every compliant SSE parser, including the
            # one in cptr, which skips anything not starting with "data:".
            yield f": switchboard preparing a model ({waited:.0f}s)\n\n".encode()

        handed_off = True
        try:
            plan = await route_task
        except Exception as error:  # noqa: BLE001
            logger.exception("routing failed")
            yield _sse_error(f"routing failed: {type(error).__name__}: {error}")
            return

        if not recognised and requested:
            plan.decision.notes.append(f"unknown model {requested!r}; routed automatically")

        # Closed explicitly so that a disconnect finalises the inner stream - its
        # route row and its in-flight reservation - now, rather than whenever the
        # abandoned generator happens to be collected.
        async with contextlib.aclosing(_stream_response(router, plan, body)) as stream:
            async for chunk in stream:
                yield chunk
    finally:
        if not handed_off:
            await _abandon_routing(router, route_task)


async def _abandon_routing(router: Router, route_task: "asyncio.Future[RoutedPlan]") -> None:
    """Stop routing whose client has gone.

    Nothing will consume the plan, so a model load must not carry on for nobody.
    The task is awaited as well as cancelled so that it has actually finished by
    the time the stream is closed, rather than merely been asked to stop.
    """
    route_task.cancel()
    try:
        plan = await route_task
    except asyncio.CancelledError:
        if not route_task.cancelled():
            # The cancellation is this task's own, not the routing's: pass it on.
            raise
        return
    except Exception as error:  # noqa: BLE001 - there is no client left to tell
        logger.warning("routing failed after the client disconnected: %s", type(error).__name__)
        return
    # Routing finished before the cancellation could land, so its row exists
    # and would otherwise never be closed.
    router.record_completion(plan.route_id, error="client disconnected")


def _sse_error(message: str, *, kind: str = "upstream_error") -> bytes:
    payload = json.dumps(_error_body(message, kind))
    return f"data: {payload}\n\ndata: [DONE]\n\n".encode()
