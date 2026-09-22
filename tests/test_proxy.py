"""The OpenAI-compatible surface.

Everything here runs the real FastAPI app over ``httpx.ASGITransport``, with a
fake llama-server behind it. That matters because the claims being checked are
about wire behaviour rather than about Python objects: does a stream arrive in
pieces, do tool definitions survive the hop, does a client that names a model
get that model.

The awkward one is streaming. A test that only checks the final body would pass
just as happily against an implementation that buffered the whole completion and
sent it at the end, which is exactly the bug worth preventing. It cannot be
caught through ``ASGITransport`` either, because that collects the entire
response before returning it, so the incremental check drives the proxy's stream
generator directly and blocks the fake server mid-stream. The end-to-end version
of that proof needs a real socket and lives in the live validation, not here.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator

import httpx
import pytest

from switchboard.app import create_app
from switchboard.config import Config
from switchboard.decisions.chain import DecisionChain
from switchboard.decisions.rules import RulesBackend
from switchboard.proxy import UsageWatcher, build_task_context, session_id_for
from switchboard.router import Router
from switchboard.store import Store
from switchboard.upstream import LlamaClient

from conftest import DEEP, FAST, FakeExecutor, FakeLlama, FixedHostFacts, default_completion


class Harness:
    """The app, wired to a fake llama-server and a fake executor."""

    def __init__(self, config: Config, llama: FakeLlama, executor: FakeExecutor) -> None:
        self.config = config
        self.llama = llama
        self.executor = executor
        self.app = create_app(config)

    async def __aenter__(self) -> "Harness":
        # The app's own lifespan is skipped: it would open a real upstream client
        # and start the MCP session manager, whose task scope does not survive
        # pytest's fixture teardown. Everything the request path reads lives on
        # app.state.router, which is supplied here instead.
        replacement = Router(
            self.config,
            upstream=LlamaClient(self.config, client=self.llama.client()),
            chain=DecisionChain(self.config, [RulesBackend(self.config)]),
            executor=self.executor,
            store=Store(self.config.db_file),
            host=FixedHostFacts(),
        )
        self.app.state.router = replacement
        self.router = replacement

        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app), base_url="http://switchboard.test"
        )
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.client.aclose()
        await self.router.chain.aclose()
        await self.router.upstream.aclose()
        if self.router.store:
            self.router.store.close()


@pytest.fixture
async def harness(config: Config) -> AsyncIterator[Harness]:
    llama = FakeLlama(resident=FAST)
    async with Harness(config, llama, FakeExecutor()) as ready:
        yield ready


DONE_FRAME = b"data: [DONE]\n\n"


def sse_chunk(text: str) -> bytes:
    """One SSE frame carrying a content delta, shaped like llama.cpp's."""
    payload = json.dumps({"choices": [{"index": 0, "delta": {"content": text}}]})
    return f"data: {payload}\n\n".encode()


def chat_body(**overrides: Any) -> dict[str, Any]:
    body = {
        "model": "switchboard-auto",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Write a Python function that reverses a string."},
        ],
    }
    body.update(overrides)
    return body


# --- model listing ----------------------------------------------------------


async def test_models_are_listed_with_and_without_the_v1_prefix(harness: Harness) -> None:
    for path in ("/v1/models", "/models"):
        response = await harness.client.get(path)
        assert response.status_code == 200
        payload = response.json()
        assert payload["object"] == "list"
        ids = [entry["id"] for entry in payload["data"]]
        assert "switchboard-auto" in ids
        assert {"fast-9b", "deep-27b", "med-27b"} <= set(ids)

        virtual = next(entry for entry in payload["data"] if entry["id"] == "switchboard-auto")
        assert virtual["switchboard"]["virtual"] is True


# --- non-streaming ----------------------------------------------------------


async def test_a_completion_is_returned_unchanged(harness: Harness) -> None:
    response = await harness.client.post("/v1/chat/completions", json=chat_body())

    assert response.status_code == 200
    assert response.json() == default_completion()
    assert response.headers["x-switchboard-model"] == "fast-9b"
    assert response.headers["x-switchboard-backend"] == "rules"
    assert response.headers["x-switchboard-reason"]
    # Headers must survive latin-1 encoding, which is all HTTP promises.
    response.headers["x-switchboard-reason"].encode("latin-1")


async def test_cptr_style_path_without_v1_also_works(harness: Harness) -> None:
    response = await harness.client.post("/chat/completions", json=chat_body())
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"]


async def test_tool_definitions_survive_the_hop(harness: Harness) -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "run_command",
                "description": "Run a shell command",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        }
    ]
    response = await harness.client.post(
        "/v1/chat/completions", json=chat_body(tools=tools, tool_choice="auto")
    )

    assert response.status_code == 200
    forwarded = harness.llama.requests[-1]
    assert forwarded["tools"] == tools, "tool definitions must reach llama.cpp untouched"
    assert forwarded["tool_choice"] == "auto"


async def test_unknown_request_fields_are_forwarded(harness: Harness) -> None:
    await harness.client.post(
        "/v1/chat/completions",
        json=chat_body(temperature=0.2, response_format={"type": "json_object"}, seed=7),
    )
    forwarded = harness.llama.requests[-1]
    assert forwarded["temperature"] == 0.2
    assert forwarded["response_format"] == {"type": "json_object"}
    assert forwarded["seed"] == 7


async def test_only_the_model_field_is_rewritten(harness: Harness) -> None:
    await harness.client.post("/v1/chat/completions", json=chat_body())
    forwarded = harness.llama.requests[-1]
    assert forwarded["model"] != "switchboard-auto"
    assert forwarded["messages"] == chat_body()["messages"]


# --- explicit selection -----------------------------------------------------


async def test_an_explicit_resident_model_does_not_switch(harness: Harness) -> None:
    response = await harness.client.post("/v1/chat/completions", json=chat_body(model="fast-9b"))

    assert response.status_code == 200
    assert response.headers["x-switchboard-model"] == "fast-9b"
    assert harness.executor.calls == []


async def test_an_explicit_other_model_switches(config: Config) -> None:
    llama = FakeLlama(resident=FAST)
    executor = FakeExecutor(load_ms=800.0)
    executor.on_switch = lambda model_id: setattr(llama, "resident", DEEP)
    async with Harness(config, llama, executor) as harness:
        response = await harness.client.post(
            "/v1/chat/completions", json=chat_body(model="deep-27b")
        )
        assert response.status_code == 200
        assert executor.calls == ["deep-27b"]
        assert response.headers["x-switchboard-model"] == "deep-27b"
        assert response.headers["x-switchboard-switched"] == "true"


async def test_an_unknown_model_is_rejected(harness: Harness) -> None:
    """A name the client chose is not re-routed: that would be a substitution.

    The answer is the one the OpenAI API gives, so SDKs surface it properly, and
    it comes before routing: nothing is decided, recorded or forwarded.
    """
    response = await harness.client.post("/v1/chat/completions", json=chat_body(model="gpt-4o"))

    assert response.status_code == 404
    error = response.json()["error"]
    assert error["code"] == "model_not_found"
    assert error["type"] == "invalid_request_error"
    assert "gpt-4o" in error["message"]
    assert harness.llama.requests == []
    assert harness.router.store.recent_routes() == []


async def test_a_request_without_a_model_is_routed_automatically(harness: Harness) -> None:
    body = chat_body()
    del body["model"]
    response = await harness.client.post("/v1/chat/completions", json=body)

    assert response.status_code == 200
    assert response.headers["x-switchboard-model"] in {"fast-9b", "deep-27b", "med-27b"}
    assert len(harness.llama.requests) == 1


async def test_a_streamed_explicit_model_that_cannot_load_is_a_409(config: Config) -> None:
    """Before the stream starts the refusal can still be a real status code."""
    llama = FakeLlama(resident=FAST)
    async with Harness(config, llama, FakeExecutor(ok=False)) as harness:
        response = await harness.client.post(
            "/v1/chat/completions", json=chat_body(model="deep-27b", stream=True)
        )

        assert response.status_code == 409
        error = response.json()["error"]
        assert error["type"] == "model_unavailable"
        assert "deep-27b" in error["message"]
        assert "reverses a string" not in error["message"]
        assert response.headers["x-switchboard-route-id"]
        assert llama.requests == []
        row = harness.router.store.recent_routes(limit=1)[0]
        assert row["downstream_status"] == 409
        assert row["error"] == "requested model deep-27b could not be loaded"


async def test_a_keepalive_stream_reports_an_unloadable_explicit_model_in_band(
    config: Config,
) -> None:
    """After keepalives the 200 is already sent, so the refusal is an SSE frame."""
    from switchboard.proxy import _stream_with_keepalive

    llama = FakeLlama(resident=FAST)
    async with Harness(config, llama, FakeExecutor(ok=False)) as harness:
        deep = harness.router.registry().by_id("deep-27b")
        plan = await harness.router.route(
            build_task_context(chat_body(), chars_per_token=3.6, session_id=None),
            explicit_model=deep,
            requested_model="deep-27b",
            streaming=True,
        )
        routed: asyncio.Future = asyncio.get_running_loop().create_future()
        routed.set_result(plan)

        request = chat_body(model="deep-27b", stream=True)
        stream = _stream_with_keepalive(harness.router, routed, request, "deep-27b", True)
        body = b"".join([chunk async for chunk in stream])

        frame = json.loads(body.split(b"\n\n")[0].removeprefix(b"data: "))
        assert frame["error"]["type"] == "model_unavailable"
        assert body.endswith(DONE_FRAME)
        assert llama.requests == []
        assert harness.router.store.get_route(plan.route_id)["downstream_status"] == 409


# --- streaming --------------------------------------------------------------


async def test_the_stream_generator_yields_as_chunks_arrive(config: Config) -> None:
    """The anti-buffering test, against the generator the proxy actually uses.

    It cannot go through httpx.ASGITransport: that collects the whole response
    body before returning it, so every stream would look buffered no matter what
    the code did. The generator is the thing that must not buffer, so the test
    drives it directly: the fake server sends one chunk and then blocks, and the
    first chunk has to arrive while it is still blocked.
    """
    from switchboard.proxy import _stream_response

    llama = FakeLlama(resident=FAST)
    llama.stream_gate = asyncio.Event()
    llama.stream_chunks = [sse_chunk(f"tok{index}") for index in range(4)] + [DONE_FRAME]

    async with Harness(config, llama, FakeExecutor()) as harness:
        plan = await harness.router.route(
            build_task_context(chat_body(), chars_per_token=3.6, session_id=None),
            streaming=True,
            source="proxy",
        )
        generator = _stream_response(harness.router, plan, chat_body(stream=True))

        first = await asyncio.wait_for(anext(generator), timeout=5.0)
        assert b"tok0" in first
        assert not llama.stream_gate.is_set(), "delivered before the server was released"

        llama.stream_gate.set()
        rest = b""
        async for chunk in generator:
            rest += chunk

    assert first + rest == b"".join(llama.stream_chunks)


async def test_a_client_that_disconnects_mid_stream_is_still_recorded(config: Config) -> None:
    """Closing the stream early still closes the route row and the reservation.

    Starlette ends an abandoned stream by closing or cancelling its generator, so
    the recording has to survive that rather than sit after the last yield.
    """
    from switchboard.proxy import _stream_response

    llama = FakeLlama(resident=FAST)
    llama.stream_gate = asyncio.Event()
    llama.stream_chunks = [sse_chunk(f"tok{index}") for index in range(4)] + [DONE_FRAME]

    async with Harness(config, llama, FakeExecutor()) as harness:
        plan = await harness.router.route(
            build_task_context(chat_body(), chars_per_token=3.6, session_id=None),
            streaming=True,
            source="proxy",
        )
        generator = _stream_response(harness.router, plan, chat_body(stream=True))
        first = await asyncio.wait_for(anext(generator), timeout=5.0)
        assert b"tok0" in first

        await generator.aclose()

        assert harness.router.inflight_count == 0
        row = harness.router.store.get_route(plan.route_id)
        assert row["error"] == "client disconnected"
        assert row["downstream_status"] == 200
        assert row["tokens_out"] == 1, "what was relayed before the disconnect still counts"


async def test_a_non_streaming_request_is_not_sent_after_the_model_changed(config: Config) -> None:
    """Residency is re-read under the reservation, not trusted from routing."""
    from switchboard.proxy import _complete

    llama = FakeLlama(resident=FAST)
    async with Harness(config, llama, FakeExecutor()) as harness:
        plan = await harness.router.route(
            build_task_context(chat_body(), chars_per_token=3.6, session_id=None),
            source="proxy",
        )
        assert plan.model_id == "fast-9b"
        llama.resident = DEEP

        response = await _complete(harness.router, plan, chat_body())

        assert response.status_code == 409
        assert b"deep-27b" in response.body
        assert llama.requests == []
        assert harness.router.inflight_count == 0


async def test_a_stream_is_relayed_byte_for_byte(config: Config) -> None:
    llama = FakeLlama(resident=FAST)
    llama.stream_chunks = [sse_chunk("alpha"), sse_chunk("beta"), DONE_FRAME]

    async with Harness(config, llama, FakeExecutor()) as harness:
        async with harness.client.stream(
            "POST", "/v1/chat/completions", json=chat_body(stream=True)
        ) as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            body = b""
            async for chunk in response.aiter_raw():
                body += chunk

    assert body == b"".join(llama.stream_chunks)


async def test_streaming_usage_is_recorded(config: Config) -> None:
    llama = FakeLlama(resident=FAST)
    async with Harness(config, llama, FakeExecutor()) as harness:
        async with harness.client.stream(
            "POST", "/v1/chat/completions", json=chat_body(stream=True)
        ) as response:
            async for _ in response.aiter_raw():
                pass

        row = harness.router.store.recent_routes(limit=1)[0]
        assert row["streaming"] is True
        assert row["downstream_status"] == 200
        # Three content deltas in the default fake stream.
        assert row["tokens_out"] == 3


# --- failures ---------------------------------------------------------------


async def test_an_upstream_status_is_passed_through(config: Config) -> None:
    """llama.cpp's own error reaches the client, with its status intact.

    A proxy that rewrote every upstream failure as 502 would hide the reason the
    request failed. 502 is reserved for "could not reach llama-server at all".
    """
    llama = FakeLlama(resident=FAST, fail_with=500)
    async with Harness(config, llama, FakeExecutor()) as harness:
        response = await harness.client.post("/v1/chat/completions", json=chat_body())

        assert response.status_code == 500
        assert response.json()["error"]["message"] == "upstream said no"
        assert response.headers["x-switchboard-model"]
        # The client gets llama.cpp's words; the database only the status,
        # because an upstream error body can quote the prompt back.
        row = harness.router.store.recent_routes(limit=1)[0]
        assert row["error"] == "upstream returned HTTP 500"


async def test_an_unreachable_upstream_becomes_a_502(config: Config) -> None:
    llama = FakeLlama(resident=FAST)
    llama.connection_error = True
    async with Harness(config, llama, FakeExecutor()) as harness:
        response = await harness.client.post("/v1/chat/completions", json=chat_body())

        assert response.status_code == 502
        assert response.json()["error"]["type"] == "upstream_error"
        row = harness.router.store.recent_routes(limit=1)[0]
        assert row["downstream_status"] == 502
        assert row["error"] == "upstream unreachable (ConnectError)"


async def test_an_unreachable_upstream_ends_a_stream_and_records_only_its_kind(
    config: Config,
) -> None:
    llama = FakeLlama(resident=FAST)
    llama.connection_error = True
    async with Harness(config, llama, FakeExecutor()) as harness:
        response = await harness.client.post(
            "/v1/chat/completions", json=chat_body(stream=True)
        )

        assert b"upstream_error" in response.content
        assert response.content.endswith(DONE_FRAME)
        row = harness.router.store.recent_routes(limit=1)[0]
        assert row["downstream_status"] == 502
        assert row["error"] == "upstream unreachable (ConnectError)"


async def test_a_streaming_upstream_error_ends_the_stream(config: Config) -> None:
    """A client must never be left hanging on a dead upstream."""
    llama = FakeLlama(resident=FAST, fail_with=500)
    async with Harness(config, llama, FakeExecutor()) as harness:
        chunks: list[bytes] = []
        async with harness.client.stream(
            "POST", "/v1/chat/completions", json=chat_body(stream=True)
        ) as response:
            async for chunk in response.aiter_raw():
                chunks.append(chunk)

        body = b"".join(chunks)
        assert b"upstream_error" in body
        assert body.endswith(b"data: [DONE]\n\n")


async def test_a_malformed_request_is_rejected_before_routing(harness: Harness) -> None:
    response = await harness.client.post("/v1/chat/completions", json={"model": "switchboard-auto"})
    assert response.status_code == 400
    assert "messages" in response.json()["error"]["message"]
    assert harness.llama.requests == []


# --- the pieces -------------------------------------------------------------


def test_usage_watcher_counts_content_deltas() -> None:
    watcher = UsageWatcher()
    for index in range(4):
        payload = json.dumps({"choices": [{"delta": {"content": f"t{index}"}}]})
        watcher.feed(f"data: {payload}\n\n".encode())
    watcher.feed(b"data: [DONE]\n\n")

    _, tokens_out, _, _ = watcher.result()
    assert tokens_out == 4


def test_usage_watcher_prefers_real_usage() -> None:
    watcher = UsageWatcher()
    watcher.feed(b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n')
    watcher.feed(
        b'data: {"choices":[],"usage":{"prompt_tokens":11,"completion_tokens":99},'
        b'"timings":{"predicted_per_second":42.5}}\n\n'
    )

    tokens_in, tokens_out, tps, _ = watcher.result()
    assert (tokens_in, tokens_out) == (11, 99)
    assert tps == pytest.approx(42.5)


def test_usage_watcher_survives_garbage_without_growing() -> None:
    watcher = UsageWatcher()
    watcher.feed(b"data: {not json at all\n\n")
    watcher.feed(b"x" * 2_000_000)  # a line that never ends
    watcher.feed(b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n')

    assert len(watcher._buffer) < 10_000
    _, tokens_out, _, _ = watcher.result()
    assert tokens_out == 1


def test_task_context_reads_the_whole_conversation() -> None:
    body = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "an answer"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "second question"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
                ],
            },
        ],
        "tools": [{"type": "function", "function": {"name": "read_file"}}],
    }
    context = build_task_context(body, chars_per_token=3.6, session_id="abc")

    assert context.message == "second question"
    assert context.prior_message == "first question"
    assert context.system_prompt == "sys"
    assert context.tool_names == ["read_file"]
    assert context.has_images is True
    assert context.turn_index == 2
    # Sized from everything, including the tool definitions, not just the last turn.
    assert context.estimated_prompt_tokens > len("second question") / 3.6


def test_session_identity_is_stable_and_hashed() -> None:
    first = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
        ]
    }
    later = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "another turn"},
        ]
    }
    other = {"messages": [{"role": "user", "content": "a different conversation"}]}

    request = httpx.Request("POST", "http://test/")
    a = session_id_for(request, first)
    b = session_id_for(request, later)
    c = session_id_for(request, other)

    assert a == b, "the same conversation must keep its identity as turns accumulate"
    assert a != c
    assert "hello" not in (a or "")


def test_an_explicit_session_header_wins() -> None:
    request = httpx.Request(
        "POST", "http://test/", headers={"X-Switchboard-Session": "conversation-7"}
    )
    identifier = session_id_for(request, {"messages": [{"role": "user", "content": "hi"}]})

    assert identifier
    assert "conversation-7" not in identifier
