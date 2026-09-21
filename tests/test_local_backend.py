"""Tests for the System One Adapter backend over the resident llama.cpp model.

Every test drives a real `httpx.AsyncClient` through `httpx.MockTransport`, with
response bodies shaped like llama.cpp build 11041 actually answers, so the wiring
under test is the wiring that runs in production: the adapter, the custom
provider, the JSON schema round trip and the error translation.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest

from switchboard.config import Config, DecisionsConfig, LocalDecisionConfig, PrivacyConfig
from switchboard.decisions.base import BackendError, BackendFacts, StepContext, TaskContext
from switchboard.decisions.local import (
    DISCRETE_NOTE,
    STEP_QUESTIONS,
    TASK_QUESTIONS,
    LocalDecisionBackend,
)
from switchboard.decisions.schema import (
    AMBIGUITY_INSTRUCTIONS,
    DIFFICULTY_CRITERIA,
    DIFFICULTY_INSTRUCTIONS,
    DOMAIN_CRITERIA,
    DOMAIN_INSTRUCTIONS,
    STEP_OUTCOME_CRITERIA,
    STEP_OUTCOME_INSTRUCTIONS,
    TOOL_NEED_INSTRUCTIONS,
)

READY = BackendFacts(
    resident_model_id="Qwen3.5-9B-Q4_K_M.gguf",
    resident_gen_tps=38.0,
    server_healthy=True,
    server_busy=False,
)

DISCRETE_TASK_ANSWERS = {
    "domain": "coding",
    "difficulty": 2,
    "tool_need": True,
    "ambiguity": False,
}

DISCRETE_STEP_ANSWERS = {
    "outcome": "partial",
    "progress": True,
    "complete": False,
    "needs_new_approach": False,
}


def make_config(**local: Any) -> Config:
    """A config whose only interesting part is `[decisions.local]`."""
    return Config(
        decisions=DecisionsConfig(local=LocalDecisionConfig(**local)),
        privacy=PrivacyConfig(max_state_chars=1500),
    )


def llama_reply(answers: dict[str, Any], *, prompt_tokens: int = 842) -> dict[str, Any]:
    """A non-streaming llama.cpp chat completion carrying a schema-valid answer."""
    content = json.dumps({"answers": answers})
    return {
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "created": 1758300000,
        "model": "Qwen3.5-9B-Q4_K_M.gguf",
        "object": "chat.completion",
        "usage": {
            "completion_tokens": 37,
            "prompt_tokens": prompt_tokens,
            "total_tokens": prompt_tokens + 37,
        },
        "timings": {
            "prompt_n": prompt_tokens,
            "prompt_per_second": 1483.2,
            "predicted_n": 37,
            "predicted_per_second": 39.6,
        },
    }


class Recorder:
    """Captures what the backend posted while answering with a canned body."""

    def __init__(self, responder: Callable[[httpx.Request], httpx.Response]) -> None:
        self._responder = responder
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self._responder(request)

    @property
    def body(self) -> dict[str, Any]:
        return json.loads(self.requests[-1].content)

    @property
    def document(self) -> str:
        return str(self.body["messages"][-1]["content"])


def replying(payload: dict[str, Any], status: int = 200) -> Recorder:
    return Recorder(lambda request: httpx.Response(status, json=payload))


def raising(error: Exception) -> Recorder:
    def responder(request: httpx.Request) -> httpx.Response:
        raise error

    return Recorder(responder)


@asynccontextmanager
async def backend_for(
    recorder: Recorder, config: Config | None = None
) -> AsyncIterator[LocalDecisionBackend]:
    client = httpx.AsyncClient(transport=httpx.MockTransport(recorder))
    backend = LocalDecisionBackend(config or make_config(), client=client)
    try:
        yield backend
    finally:
        await backend.aclose()
        await client.aclose()


# --- the questions ----------------------------------------------------------


def test_questions_are_the_shared_schema_questions_verbatim() -> None:
    """The point of the shared constants is that two backends answer one question."""
    domain = TASK_QUESTIONS["domain"]
    assert domain.instructions == DOMAIN_INSTRUCTIONS
    assert dict(domain.criteria) == DOMAIN_CRITERIA

    difficulty = TASK_QUESTIONS["difficulty"]
    assert difficulty.instructions == DIFFICULTY_INSTRUCTIONS
    assert list(difficulty.criteria) == DIFFICULTY_CRITERIA

    assert TASK_QUESTIONS["tool_need"].instructions == TOOL_NEED_INSTRUCTIONS
    assert TASK_QUESTIONS["ambiguity"].instructions == AMBIGUITY_INSTRUCTIONS

    outcome = STEP_QUESTIONS["outcome"]
    assert outcome.instructions == STEP_OUTCOME_INSTRUCTIONS
    assert dict(outcome.criteria) == STEP_OUTCOME_CRITERIA


# --- happy path -------------------------------------------------------------


async def test_decide_task_parses_a_llama_cpp_answer() -> None:
    recorder = replying(llama_reply(DISCRETE_TASK_ANSWERS))
    async with backend_for(recorder) as backend:
        decision = await backend.decide_task(
            TaskContext(message="my pytest fixture leaks a file handle, why?"), READY
        )

    assert decision.domain == "coding"
    assert decision.difficulty == pytest.approx(2.0)
    assert decision.difficulty_level == 2
    assert decision.tool_need == pytest.approx(1.0)
    assert decision.ambiguity == pytest.approx(0.0)
    assert decision.backend == "local"
    assert decision.degraded is False
    assert decision.usage == {"input_tokens": 842, "output_tokens": 37}
    assert decision.latency_ms > 0

    assert recorder.requests[-1].url.path == "/v1/chat/completions"
    assert str(recorder.requests[-1].url).startswith("http://127.0.0.1:10000")


async def test_discrete_mode_reports_no_probabilities_and_no_confidence() -> None:
    """A one-hot distribution is an artefact of the mode; it must not be reported."""
    recorder = replying(llama_reply(DISCRETE_TASK_ANSWERS))
    async with backend_for(recorder) as backend:
        decision = await backend.decide_task(TaskContext(message="rename this variable"), READY)

    assert decision.domain_probabilities == {}
    assert decision.difficulty_probabilities == {}
    assert decision.domain_confidence is None
    assert decision.difficulty_confidence is None
    assert DISCRETE_NOTE in decision.notes


async def test_probabilities_mode_fills_the_distributions() -> None:
    answers = {
        "domain": {
            "general": 0.05,
            "coding": 0.75,
            "medical": 0.0,
            "writing": 0.05,
            "research": 0.15,
            "system_ops": 0.0,
        },
        "difficulty": {"0": 0.0, "1": 0.2, "2": 0.6, "3": 0.2},
        "tool_need": 0.8,
        "ambiguity": 0.1,
    }
    recorder = replying(llama_reply(answers))
    async with backend_for(recorder, make_config(answer_mode="probabilities")) as backend:
        decision = await backend.decide_task(TaskContext(message="compare two libraries"), READY)

    assert decision.domain == "coding"
    assert decision.domain_probabilities["coding"] == pytest.approx(0.75)
    assert decision.domain_confidence is not None and decision.domain_confidence > 0.5
    assert decision.difficulty == pytest.approx(2.0)
    assert decision.difficulty_probabilities == {
        "0": pytest.approx(0.0),
        "1": pytest.approx(0.2),
        "2": pytest.approx(0.6),
        "3": pytest.approx(0.2),
    }
    assert decision.difficulty_confidence is not None
    assert decision.tool_need == pytest.approx(0.8)
    assert decision.ambiguity == pytest.approx(0.1)
    assert DISCRETE_NOTE not in decision.notes


async def test_decide_step_parses_a_llama_cpp_answer() -> None:
    recorder = replying(llama_reply(DISCRETE_STEP_ANSWERS))
    async with backend_for(recorder) as backend:
        decision = await backend.decide_step(
            StepContext(
                goal="get the integration suite green",
                action="uv run pytest tests/test_store.py",
                result="3 passed, 1 failed: test_prune_removes_old_rows",
                exit_code=1,
                attempt=2,
            ),
            READY,
        )

    assert decision.outcome == "partial"
    assert decision.progress == pytest.approx(1.0)
    assert decision.complete == pytest.approx(0.0)
    assert decision.needs_new_approach == pytest.approx(0.0)
    assert decision.backend == "local"
    assert decision.outcome_probabilities == {}
    assert decision.outcome_confidence is None

    document = recorder.document
    assert "uv run pytest tests/test_store.py" in document
    assert "test_prune_removes_old_rows" in document
    # The questions name both paths, so both have to be present in the document.
    assert "task" in document and "step" in document


# --- the request body -------------------------------------------------------


async def test_request_disables_thinking_and_pins_determinism() -> None:
    """The resident Qwen3.5 models reason by default; a decision cannot afford it."""
    recorder = replying(llama_reply(DISCRETE_TASK_ANSWERS))
    config = make_config(max_output_tokens=256)
    async with backend_for(recorder, config) as backend:
        await backend.decide_task(TaskContext(message="hello"), READY)

    body = recorder.body
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["reasoning_effort"] == "none"
    assert body["temperature"] == 0.0
    assert body["max_tokens"] == 256
    assert body["cache_prompt"] is True
    assert body["model"] == READY.resident_model_id


async def test_request_asks_for_a_json_schema_response_format() -> None:
    recorder = replying(llama_reply(DISCRETE_TASK_ANSWERS))
    async with backend_for(recorder) as backend:
        await backend.decide_task(TaskContext(message="hello"), READY)

    response_format = recorder.body["response_format"]
    assert response_format["type"] == "json_schema"
    schema = response_format["json_schema"]["schema"]
    answers = schema["$defs"]["TypeSafeAnswers"]["properties"]
    assert set(answers) == {"domain", "difficulty", "tool_need", "ambiguity"}


async def test_request_carries_the_configured_timeout() -> None:
    """The timeout has to abort the HTTP call, not just the coroutine waiting on it."""
    recorder = replying(llama_reply(DISCRETE_TASK_ANSWERS))
    async with backend_for(recorder, make_config(timeout_s=7.5)) as backend:
        await backend.decide_task(TaskContext(message="hello"), READY)

    timeout = recorder.requests[-1].extensions["timeout"]
    assert timeout["read"] == pytest.approx(7.5)
    assert timeout["connect"] == pytest.approx(5.0)


async def test_state_is_truncated_but_not_redacted() -> None:
    """Local text never leaves the machine, so redaction would cost signal for nothing."""
    secret = "sk-liveXXXXXXXXXXXXXXXXXXXXXXXX"
    message = f"my deploy script exports OPENAI_API_KEY={secret} " + ("filler word " * 400)
    recorder = replying(llama_reply(DISCRETE_TASK_ANSWERS))
    async with backend_for(recorder) as backend:
        await backend.decide_task(TaskContext(message=message), READY)

    document = recorder.document
    assert secret in document
    assert "REDACTED" not in document
    assert "chars omitted" in document
    assert len(document) < len(message)


# --- availability -----------------------------------------------------------


async def test_availability_ready_when_the_stack_is_idle_and_fast() -> None:
    recorder = replying(llama_reply(DISCRETE_TASK_ANSWERS))
    async with backend_for(recorder) as backend:
        availability = await backend.availability(READY)
    assert availability.ready
    assert availability.state == "ready"


async def test_availability_refuses_an_unhealthy_server_with_a_reason() -> None:
    recorder = replying(llama_reply(DISCRETE_TASK_ANSWERS))
    async with backend_for(recorder) as backend:
        availability = await backend.availability(BackendFacts(server_healthy=False))
    assert availability.state == "unavailable"
    assert "127.0.0.1:10000" in availability.reason
    assert "/health" in availability.reason


async def test_availability_refuses_a_busy_server_because_parallel_is_one() -> None:
    recorder = replying(llama_reply(DISCRETE_TASK_ANSWERS))
    facts = BackendFacts(server_healthy=True, server_busy=True, resident_gen_tps=38.0)
    async with backend_for(recorder) as backend:
        availability = await backend.availability(facts)
    assert availability.state == "unavailable"
    assert "--parallel 1" in availability.reason

    # ...unless the operator turned that guard off.
    async with backend_for(recorder, make_config(skip_when_busy=False)) as permissive:
        assert (await permissive.availability(facts)).ready


async def test_availability_refuses_a_model_that_generates_too_slowly() -> None:
    facts = BackendFacts(
        server_healthy=True, resident_model_id="MedGemma-27B.gguf", resident_gen_tps=5.0
    )
    recorder = replying(llama_reply(DISCRETE_TASK_ANSWERS))
    async with backend_for(recorder, make_config(min_resident_gen_tps=15.0)) as backend:
        availability = await backend.availability(facts)
    assert availability.state == "unavailable"
    assert "5.0 tok/s" in availability.reason
    assert "15.0 tok/s" in availability.reason
    assert "MedGemma-27B.gguf" in availability.reason


async def test_availability_allows_an_unmeasured_model() -> None:
    """Refusing on a number nobody has measured would keep the backend off forever."""
    facts = BackendFacts(server_healthy=True, resident_gen_tps=None)
    recorder = replying(llama_reply(DISCRETE_TASK_ANSWERS))
    async with backend_for(recorder, make_config(min_resident_gen_tps=15.0)) as backend:
        assert (await backend.availability(facts)).ready


async def test_availability_is_unconfigured_when_disabled() -> None:
    recorder = replying(llama_reply(DISCRETE_TASK_ANSWERS))
    async with backend_for(recorder, make_config(enabled=False)) as backend:
        availability = await backend.availability(READY)
    assert availability.state == "unconfigured"
    assert "decisions.local.enabled" in availability.reason


# --- failures ---------------------------------------------------------------


async def test_server_error_becomes_a_retryable_backend_error() -> None:
    recorder = replying({"error": {"message": "slot unavailable"}}, status=500)
    async with backend_for(recorder) as backend:
        with pytest.raises(BackendError) as caught:
            await backend.decide_task(TaskContext(message="hello"), READY)

    error = caught.value
    assert error.backend == "local"
    assert error.kind == "server"
    assert error.retryable is True
    assert error.counts_against_health is True


async def test_bad_request_becomes_a_non_retryable_backend_error() -> None:
    recorder = replying({"error": {"message": "unsupported response_format"}}, status=400)
    async with backend_for(recorder) as backend:
        with pytest.raises(BackendError) as caught:
            await backend.decide_task(TaskContext(message="hello"), READY)

    assert caught.value.kind == "invalid"
    assert caught.value.retryable is False


async def test_timeout_becomes_a_timeout_backend_error() -> None:
    recorder = raising(httpx.ReadTimeout("timed out"))
    async with backend_for(recorder) as backend:
        with pytest.raises(BackendError) as caught:
            await backend.decide_task(TaskContext(message="hello"), READY)

    assert caught.value.kind == "timeout"
    assert caught.value.counts_against_health is True


async def test_connection_failure_becomes_a_connection_backend_error() -> None:
    recorder = raising(httpx.ConnectError("connection refused"))
    async with backend_for(recorder) as backend:
        with pytest.raises(BackendError) as caught:
            await backend.decide_step(
                StepContext(goal="g", action="a", result="r"), READY
            )

    assert caught.value.kind == "connection"


async def test_an_answer_outside_the_schema_becomes_an_invalid_backend_error() -> None:
    """A half-filled decision must never reach the policy as if it were real."""
    recorder = replying(llama_reply({"domain": "coding", "difficulty": 2}))
    async with backend_for(recorder) as backend:
        with pytest.raises(BackendError) as caught:
            await backend.decide_task(TaskContext(message="hello"), READY)

    assert caught.value.kind == "invalid"
    assert caught.value.retryable is False


async def test_an_empty_completion_becomes_a_backend_error() -> None:
    recorder = replying(
        {
            "choices": [{"index": 0, "message": {"role": "assistant", "content": ""}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 0},
        }
    )
    async with backend_for(recorder) as backend:
        with pytest.raises(BackendError) as caught:
            await backend.decide_task(TaskContext(message="hello"), READY)

    assert caught.value.backend == "local"


async def test_error_messages_never_carry_the_prompt() -> None:
    """The adapter hangs the full attempt trace off the error; none of it may escape."""
    message = "the quick brown fox jumps over the lazy dog"
    recorder = replying({"error": {"message": "slot unavailable"}}, status=500)
    async with backend_for(recorder) as backend:
        with pytest.raises(BackendError) as caught:
            await backend.decide_task(TaskContext(message=message), READY)

    assert message not in str(caught.value)
