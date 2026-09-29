"""Shared fixtures: a whole fake stack on disk and a fake llama-server.

Two decisions shape this file.

**The stack is built, not mocked.** ``build_registry`` reads
``Config\\models.json``, the GGUF headers and the capability overlay, so the
tests write all three into ``tmp_path`` - including *real* GGUF files, byte for
byte, produced by ``write_gguf``. That keeps the registry, the header reader and
the policy on their production code paths, and it means a test can say "this
model's template has no tool support" by writing a template without tools rather
than by patching an attribute.

**llama-server is an ``httpx.MockTransport``.** ``FakeLlama`` answers /health,
/v1/models, /props, /slots and /v1/chat/completions, streaming or not, and
records every request body it received. No test needs a network, a GPU, or the
real stack, so the suite passes with everything on this machine stopped.
"""

from __future__ import annotations

import asyncio
import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import httpx
import pytest

from switchboard.config import Config
from switchboard.decisions.chain import DecisionChain
from switchboard.decisions.rules import RulesBackend
from switchboard.executor import SwitchResult
from switchboard.registry import ModelEntry, RegistrySnapshot
from switchboard.router import Router
from switchboard.store import Store
from switchboard.system import HostFacts, MemoryStatus
from switchboard.upstream import LlamaClient

# --- GGUF writing -----------------------------------------------------------

_GGUF_UINT32 = 4
_GGUF_UINT64 = 10
_GGUF_STRING = 8


def _gguf_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def write_gguf(
    path: Path,
    *,
    architecture: str = "qwen35",
    context_length: int = 262144,
    block_count: int = 32,
    chat_template: str = "{% for message in messages %}{{ message.content }}{% endfor %}",
    name: str | None = None,
    padding: int = 0,
) -> Path:
    """Write a real, minimal GGUF file that `switchboard.gguf` can read."""
    entries: list[tuple[str, int, bytes]] = [
        ("general.architecture", _GGUF_STRING, _gguf_string(architecture)),
        ("general.name", _GGUF_STRING, _gguf_string(name or path.stem)),
        (f"{architecture}.context_length", _GGUF_UINT32, struct.pack("<I", context_length)),
        (f"{architecture}.block_count", _GGUF_UINT32, struct.pack("<I", block_count)),
        ("tokenizer.chat_template", _GGUF_STRING, _gguf_string(chat_template)),
    ]
    body = b""
    for key, value_type, payload in entries:
        body += _gguf_string(key) + struct.pack("<I", value_type) + payload

    header = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", len(entries))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + body + (b"\0" * padding))
    return path


# The three models mirror the real machine: a fast fully-offloaded 9B, a slower
# partially-offloaded 27B, and a medical specialist whose template has no tools.
FAST = "Fast-9B-Q6.gguf"
DEEP = "Deep-27B-IQ3.gguf"
MED = "Med-27B-Q4.gguf"

_TOOL_TEMPLATE = "{% if tools %}{{ tools }}{% endif %}{% for m in messages %}{{ m.content }}{% endfor %}"
_NO_TOOL_TEMPLATE = "{% for m in messages %}{{ m.content }}{% endfor %}"


@dataclass
class FakeStack:
    """Where the fake C:\\AI lives for one test."""

    root: Path
    models_dir: Path
    models_config: Path
    overlay: Path


@pytest.fixture
def stack(tmp_path: Path) -> FakeStack:
    root = tmp_path / "AI"
    models_dir = root / "models"
    write_gguf(models_dir / FAST, block_count=32, context_length=262144, chat_template=_TOOL_TEMPLATE)
    write_gguf(models_dir / DEEP, block_count=65, context_length=262144, chat_template=_TOOL_TEMPLATE)
    write_gguf(
        models_dir / MED,
        architecture="gemma3",
        block_count=62,
        context_length=131072,
        chat_template=_NO_TOOL_TEMPLATE,
    )

    models_config = root / "Config" / "models.json"
    models_config.parent.mkdir(parents=True, exist_ok=True)
    models_config.write_text(
        json.dumps(
            {
                "activeModel": FAST,
                "models": {
                    FAST: {"contextSize": 114688, "gpuLayers": 999, "threads": 8},
                    DEEP: {"contextSize": 20480, "gpuLayers": 60, "threads": 20},
                    MED: {"contextSize": 32768, "gpuLayers": 31, "threads": 20},
                },
            }
        ),
        encoding="utf-8",
    )

    overlay = tmp_path / "config" / "models.toml"
    overlay.parent.mkdir(parents=True, exist_ok=True)
    overlay.write_text(
        f"""
[[model]]
file = "{FAST}"
id = "fast-9b"
aliases = ["fast9"]
role = "general-fast"
enabled = true
tool_calling = "good"
[model.capability]
general = 2.1
coding = 2.0
research = 1.9
medical = 1.0
writing = 2.1
system_ops = 2.0
[model.measured]
gen_tps = 30.0
gen_tps_source = "fixture"
prompt_tps = 300.0
prompt_tps_source = "fixture"
load_s = 12.0
load_s_source = "fixture"

[[model]]
file = "{DEEP}"
id = "deep-27b"
role = "deep-reasoning"
enabled = true
tool_calling = "good"
[model.capability]
general = 2.6
coding = 2.5
research = 2.5
medical = 1.6
writing = 2.5
system_ops = 2.4
[model.measured]
gen_tps = 10.0
gen_tps_source = "fixture"
prompt_tps = 60.0
prompt_tps_source = "fixture"
load_s = 22.0
load_s_source = "fixture"

[[model]]
file = "{MED}"
id = "med-27b"
role = "medical-specialist"
enabled = true
tool_calling = "none"
[model.capability]
general = 1.9
coding = 1.2
research = 1.8
medical = 2.8
writing = 1.9
system_ops = 1.0
[model.measured]
gen_tps = 5.0
gen_tps_source = "fixture"
prompt_tps = 23.0
prompt_tps_source = "fixture"
load_s = 27.6
load_s_source = "fixture"
""".strip(),
        encoding="utf-8",
    )

    return FakeStack(root=root, models_dir=models_dir, models_config=models_config, overlay=overlay)


@pytest.fixture
def config(stack: FakeStack, tmp_path: Path) -> Config:
    """A Config pointing entirely at the fake stack."""
    config = Config()
    config.stack.ai_root = stack.root
    config.stack.models_dir = stack.models_dir
    config.stack.models_config = stack.models_config
    config.stack.dashboard_url = "http://dashboard.test"
    config.models_path = stack.overlay
    config.upstream.base_url = "http://llama.test"
    config.observability.db_path = tmp_path / "switchboard.sqlite3"
    config.routing.mode = "active"
    # Cloud decisions are never attempted in tests, whatever the environment has.
    config.typesafe_api_key = None
    # Never the real Lecture Notes status file: a lecture being recorded on this
    # machine must not change what the tests route.
    config.stack.recording_status_file = tmp_path / "lecture-notes-status.json"
    return config


# --- fake llama-server ------------------------------------------------------


@dataclass
class FakeLlama:
    """An httpx.MockTransport standing in for llama-server."""

    resident: str = FAST
    healthy: bool = True
    busy: bool = False
    # Set to a status code to make chat completions fail.
    fail_with: int | None = None
    # Set to make the socket itself fail, as a stopped llama-server would.
    connection_error: bool = False
    stream_chunks: list[bytes] = field(default_factory=list)
    # When set, the stream yields its first chunk and then waits for this event
    # before sending the rest. A proxy that buffers cannot deliver anything
    # until it is set, which is what makes incremental delivery testable.
    stream_gate: asyncio.Event | None = None
    completion: dict[str, Any] | None = None
    requests: list[dict[str, Any]] = field(default_factory=list)
    paths: list[str] = field(default_factory=list)

    def model_path(self) -> str:
        return f"C:\\AI\\models\\{self.resident}"

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._handle)

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=self.transport(), base_url="http://llama.test")

    def _handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.paths.append(path)

        if path == "/health":
            if not self.healthy:
                return httpx.Response(503, json={"error": "loading"})
            return httpx.Response(200, json={"status": "ok"})

        if path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": self.model_path()}]})

        if path == "/props":
            return httpx.Response(200, json={"model_path": self.model_path(), "n_ctx": 114688})

        if path == "/slots":
            return httpx.Response(200, json=[{"id": 0, "is_processing": self.busy}])

        if path.endswith("/chat/completions"):
            body = json.loads(request.content or b"{}")
            self.requests.append(body)
            if self.connection_error:
                raise httpx.ConnectError("connection refused", request=request)
            if self.fail_with:
                return httpx.Response(self.fail_with, json={"error": {"message": "upstream said no"}})
            if body.get("stream"):
                chunks = self.stream_chunks or default_stream_chunks()
                return httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    content=_iter_bytes(chunks, gate=self.stream_gate),
                )
            return httpx.Response(200, json=self.completion or default_completion())

        return httpx.Response(404, json={"error": f"no route {path}"})


def _iter_bytes(chunks: Iterable[bytes], *, gate: asyncio.Event | None = None):
    """An async byte stream: httpx requires one for an async transport."""

    async def generator():
        for index, chunk in enumerate(chunks):
            if gate is not None and index == 1:
                # Everything after the first chunk waits for the test's say-so.
                await gate.wait()
            await asyncio.sleep(0)
            yield chunk

    return generator()


def default_completion(content: str = "Hello from the fake model.") -> dict[str, Any]:
    return {
        "id": "chatcmpl-fake",
        "object": "chat.completion",
        "model": "fake",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 41, "completion_tokens": 11, "total_tokens": 52},
        "timings": {"predicted_per_second": 28.5, "prompt_per_second": 310.0, "predicted_n": 11},
    }


def default_stream_chunks(words: tuple[str, ...] = ("one", " two", " three")) -> list[bytes]:
    chunks = []
    for word in words:
        payload = {
            "id": "chatcmpl-fake",
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"content": word}, "finish_reason": None}],
        }
        chunks.append(f"data: {json.dumps(payload)}\n\n".encode())
    chunks.append(b"data: [DONE]\n\n")
    return chunks


# --- fakes for the pieces the router drives ---------------------------------


class FakeExecutor:
    """Records switches instead of restarting anything."""

    name = "fake"

    def __init__(self, *, available: bool = True, ok: bool = True, load_ms: float = 12000.0) -> None:
        self._available = available
        self._ok = ok
        self._load_ms = load_ms
        self.calls: list[str] = []
        self.on_switch: Callable[[str], None] | None = None

    async def available(self) -> tuple[bool, str]:
        return self._available, "fake executor" if self._available else "fake executor is off"

    async def switch(self, target: ModelEntry, *, from_model: str | None) -> SwitchResult:
        self.calls.append(target.id)
        if self.on_switch:
            self.on_switch(target.id)
        return SwitchResult(
            ok=self._ok,
            executor=self.name,
            from_model=from_model,
            to_model=target.id,
            load_ms=self._load_ms if self._ok else None,
            error=None if self._ok else "fake executor refused",
        )


class FixedHostFacts(HostFacts):
    """Host facts with the numbers a test wants, and no subprocess."""

    def __init__(self, *, free_gb: float = 8.0, total_gb: float = 15.4) -> None:
        super().__init__(ttl_s=3600)
        self._fixed = MemoryStatus(total_gb=total_gb, free_gb=free_gb)

    def memory(self, *, force: bool = False) -> MemoryStatus:
        return self._fixed

    async def gpu(self):  # type: ignore[override]
        from switchboard.system import GpuStatus

        return GpuStatus(name="fake", total_mb=12227, used_mb=0, free_mb=12227, driver="0")


def build_router(
    config: Config,
    llama: FakeLlama,
    *,
    executor: Any | None = None,
    store: Store | None = None,
    free_ram_gb: float = 8.0,
) -> Router:
    """A real Router over real rules, a real store and a fake llama-server."""
    upstream = LlamaClient(config, client=llama.client())
    chain = DecisionChain(config, [RulesBackend(config)])
    return Router(
        config,
        upstream=upstream,
        chain=chain,
        executor=executor or FakeExecutor(),
        store=store if store is not None else Store(config.db_file),
        host=FixedHostFacts(free_gb=free_ram_gb),
    )


def snapshot_of(router: Router) -> RegistrySnapshot:
    return router.registry(refresh=True)
