"""Tests for the experimental first-token-logprob backend.

The whole value of this backend is the honesty of its numbers, so the tests
mostly check bookkeeping: which tokens count as a label, how much mass landed
somewhere that is not a label, and that none of it is quietly rounded away.
Every response body is shaped like llama.cpp build 11041's `logprobs.content[i]`,
served through `httpx.MockTransport`.
"""

from __future__ import annotations

import json
import math
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest

from switchboard.config import Config, DecisionsConfig, LogitDecisionConfig, PrivacyConfig
from switchboard.decisions.base import BackendError, BackendFacts, StepContext, TaskContext
from switchboard.decisions.logit import (
    MAX_TOP_LOGPROBS,
    UNCALIBRATED_NOTE,
    LogitDecisionBackend,
    match_label,
)

READY = BackendFacts(
    resident_model_id="Qwen3.5-9B-Q4_K_M.gguf",
    resident_gen_tps=38.0,
    server_healthy=True,
    server_busy=False,
)

Tops = Sequence[tuple[str, float]]

# One entry per probe, in the order the backend asks its questions.
TASK_TOPS: list[Tops] = [
    [(" B", 0.86), ("B", 0.05), (" the", 0.04), ("A", 0.03)],  # domain -> coding
    [("2", 0.55), ("1", 0.30), ("3", 0.10), ("0", 0.03)],  # difficulty
    [(" yes", 0.78), (" no", 0.19)],  # tool_need
    [("no", 0.91), ("yes", 0.06)],  # ambiguity
]

STEP_TOPS: list[Tops] = [
    [("B", 0.62), ("A", 0.25), ("C", 0.10)],  # outcome -> partial
    [("yes", 0.80), ("no", 0.17)],  # progress
    [("no", 0.88), ("yes", 0.09)],  # complete
    [("no", 0.70), ("yes", 0.27)],  # needs_new_approach
]


def make_config(**logit: Any) -> Config:
    settings = {"enabled": True, **logit}
    return Config(
        decisions=DecisionsConfig(logit=LogitDecisionConfig(**settings)),
        privacy=PrivacyConfig(max_state_chars=1500),
    )


def logprob_reply(tops: Tops) -> dict[str, Any]:
    """A `max_tokens=1` completion with llama.cpp's logprobs block attached."""
    entries = [
        {"token": token, "logprob": math.log(probability), "bytes": list(token.encode())}
        for token, probability in tops
    ]
    first = entries[0]
    return {
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": str(first["token"]).strip()},
                "logprobs": {
                    "content": [
                        {
                            "token": first["token"],
                            "logprob": first["logprob"],
                            "bytes": first["bytes"],
                            "top_logprobs": entries,
                        }
                    ]
                },
                "finish_reason": "length",
            }
        ],
        "model": "Qwen3.5-9B-Q4_K_M.gguf",
        "object": "chat.completion",
        "usage": {"prompt_tokens": 901, "completion_tokens": 1, "total_tokens": 902},
        "timings": {"prompt_per_second": 1402.7, "predicted_per_second": 39.1},
    }


class Recorder:
    """Answers each probe in turn while keeping every request it was sent."""

    def __init__(self, responder: Callable[[httpx.Request, int], httpx.Response]) -> None:
        self._responder = responder
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        index = len(self.requests)
        self.requests.append(request)
        return self._responder(request, index)

    def body(self, index: int = -1) -> dict[str, Any]:
        return json.loads(self.requests[index].content)

    def document(self, index: int = -1) -> str:
        return str(self.body(index)["messages"][-1]["content"])


def answering(replies: Sequence[Tops]) -> Recorder:
    return Recorder(lambda request, index: httpx.Response(200, json=logprob_reply(replies[index])))


def failing(status: int, payload: dict[str, Any]) -> Recorder:
    return Recorder(lambda request, index: httpx.Response(status, json=payload))


def raising(error: Exception) -> Recorder:
    def responder(request: httpx.Request, index: int) -> httpx.Response:
        raise error

    return Recorder(responder)


def returning(payload: dict[str, Any]) -> Recorder:
    return Recorder(lambda request, index: httpx.Response(200, json=payload))


@asynccontextmanager
async def backend_for(
    recorder: Recorder, config: Config | None = None
) -> AsyncIterator[LogitDecisionBackend]:
    client = httpx.AsyncClient(transport=httpx.MockTransport(recorder))
    backend = LogitDecisionBackend(config or make_config(), client=client)
    try:
        yield backend
    finally:
        await backend.aclose()
        await client.aclose()


# --- token matching ---------------------------------------------------------

LETTERS = ("A", "B", "C")
YES_NO = ("yes", "no")
DIGITS = ("0", "1", "2", "3")


@pytest.mark.parametrize(
    ("token", "labels", "expected", "by_fallback"),
    [
        (" B", LETTERS, "B", False),  # llama.cpp prefixes the space
        ("b", LETTERS, "B", False),  # case is the tokenizer's business, not ours
        ("B)", LETTERS, "B", True),  # punctuation stripped
        ('"yes"', YES_NO, "yes", True),
        ("YES,", YES_NO, "yes", True),
        ("ye", YES_NO, "yes", True),  # a label the tokenizer split
        ("n", YES_NO, "no", True),
        (" 3.", DIGITS, "3", True),
        ("2", DIGITS, "2", False),
        # The guard that earns the rule its keep: an ordinary word must not be
        # allowed to vote for the single-letter label it happens to start with.
        ("and", LETTERS, None, False),
        ("about", LETTERS, None, False),
        ("maybe", YES_NO, None, False),
        ("\n", LETTERS, None, False),
        ("   ", LETTERS, None, False),
        ("", LETTERS, None, False),
    ],
)
def test_match_label_follows_the_documented_rule(
    token: str, labels: tuple[str, ...], expected: str | None, by_fallback: bool
) -> None:
    assert match_label(token, labels) == (expected, by_fallback)


def test_match_label_rejects_an_ambiguous_prefix() -> None:
    """Two labels sharing a first character make the fallback unusable, not lucky."""
    assert match_label("s", ("success", "stopped")) == (None, False)


# --- happy path -------------------------------------------------------------


async def test_decide_task_reads_the_first_token_distribution() -> None:
    recorder = answering(TASK_TOPS)
    async with backend_for(recorder) as backend:
        decision = await backend.decide_task(
            TaskContext(message="why does my build script fail on Windows?"), READY
        )

    assert len(recorder.requests) == 4
    assert decision.backend == "logit"
    assert decision.domain == "coding"
    # 0.86 + 0.05 of the 0.94 mass that landed on a letter.
    assert decision.domain_probabilities["coding"] == pytest.approx(0.91 / 0.94)
    assert decision.domain_probabilities["general"] == pytest.approx(0.03 / 0.94)
    assert decision.domain_probabilities["writing"] == pytest.approx(0.0)
    assert sum(decision.domain_probabilities.values()) == pytest.approx(1.0)
    assert decision.domain_confidence is not None and decision.domain_confidence > 0.8

    # Probability-weighted level, not the modal one.
    expected_difficulty = (0 * 0.03 + 1 * 0.30 + 2 * 0.55 + 3 * 0.10) / 0.98
    assert decision.difficulty == pytest.approx(expected_difficulty)
    assert decision.difficulty_level == 2
    assert decision.difficulty_probabilities["2"] == pytest.approx(0.55 / 0.98)
    assert decision.difficulty_confidence is not None

    assert decision.tool_need == pytest.approx(0.78 / 0.97)
    assert decision.ambiguity == pytest.approx(0.06 / 0.97)


async def test_decide_step_reads_the_first_token_distribution() -> None:
    recorder = answering(STEP_TOPS)
    async with backend_for(recorder) as backend:
        decision = await backend.decide_step(
            StepContext(
                goal="make the prune command drop expired rows",
                action="uv run switchboard prune --days 30",
                result="deleted 0 rows",
                exit_code=0,
            ),
            READY,
        )

    assert decision.outcome == "partial"
    assert decision.outcome_probabilities["partial"] == pytest.approx(0.62 / 0.97)
    assert decision.outcome_probabilities["success"] == pytest.approx(0.25 / 0.97)
    assert decision.progress == pytest.approx(0.80 / 0.97)
    assert decision.complete == pytest.approx(0.09 / 0.97)
    assert decision.needs_new_approach == pytest.approx(0.27 / 0.97)
    assert decision.backend == "logit"

    document = recorder.document(0)
    assert "uv run switchboard prune --days 30" in document
    assert "task.goal" in document and "step.goal" in document


async def test_every_decision_says_the_probabilities_are_not_calibrated() -> None:
    recorder = answering(TASK_TOPS)
    async with backend_for(recorder) as backend:
        decision = await backend.decide_task(TaskContext(message="hello"), READY)

    assert decision.notes[0] == UNCALIBRATED_NOTE
    assert any(note.startswith("domain:") for note in decision.notes)
    assert any(note.startswith("difficulty:") for note in decision.notes)


# --- the request body -------------------------------------------------------


async def test_probes_ask_for_one_token_with_logprobs_and_no_thinking() -> None:
    recorder = answering(TASK_TOPS)
    async with backend_for(recorder, make_config(top_logprobs=12)) as backend:
        await backend.decide_task(TaskContext(message="hello"), READY)

    for index in range(4):
        body = recorder.body(index)
        assert body["max_tokens"] == 1
        assert body["logprobs"] is True
        assert body["top_logprobs"] == 12
        assert body["temperature"] == 0.0
        assert body["chat_template_kwargs"] == {"enable_thinking": False}
        assert body["reasoning_effort"] == "none"
        assert body["cache_prompt"] is True
        assert body["model"] == READY.resident_model_id
        assert recorder.requests[index].url.path == "/v1/chat/completions"


async def test_top_logprobs_is_clamped_to_what_llama_cpp_accepts() -> None:
    recorder = answering(TASK_TOPS)
    async with backend_for(recorder, make_config(top_logprobs=200)) as backend:
        await backend.decide_task(TaskContext(message="hello"), READY)
    assert recorder.body(0)["top_logprobs"] == MAX_TOP_LOGPROBS


async def test_probes_share_a_prompt_prefix_so_the_kv_cache_survives() -> None:
    """The document leads and the question trails, or cache_prompt buys nothing."""
    recorder = answering(TASK_TOPS)
    async with backend_for(recorder) as backend:
        await backend.decide_task(TaskContext(message="unique-marker-in-the-document"), READY)

    documents = [recorder.document(index) for index in range(4)]
    prefix = documents[0].split("</document>")[0]
    assert "unique-marker-in-the-document" in prefix
    assert all(document.startswith(prefix) for document in documents)
    assert len({document for document in documents}) == 4


async def test_the_timeout_is_a_budget_for_the_whole_decision() -> None:
    recorder = answering(TASK_TOPS)
    async with backend_for(recorder, make_config(timeout_s=2.0)) as backend:
        await backend.decide_task(TaskContext(message="hello"), READY)

    for request in recorder.requests:
        timeout = request.extensions["timeout"]
        assert 0 < timeout["read"] <= 2.0
        # The connect timeout may not outlive the budget it is spending.
        assert timeout["connect"] <= 2.0
    reads = [request.extensions["timeout"]["read"] for request in recorder.requests]
    assert reads == sorted(reads, reverse=True)


async def test_state_is_truncated_but_not_redacted() -> None:
    secret = "sk-liveXXXXXXXXXXXXXXXXXXXXXXXX"
    message = f"my script sets OPENAI_API_KEY={secret} " + ("filler word " * 400)
    recorder = answering(TASK_TOPS)
    async with backend_for(recorder) as backend:
        await backend.decide_task(TaskContext(message=message), READY)

    document = recorder.document(0)
    assert secret in document
    assert "REDACTED" not in document
    assert "chars omitted" in document


# --- invalid mass accounting ------------------------------------------------


async def test_invalid_mass_is_measured_and_reported() -> None:
    """Renormalising hides how much mass was never an answer; the diagnostics do not."""
    noisy: list[Tops] = [
        [("Sure", 0.45), (" B", 0.30), ("\n", 0.15), ("A", 0.05)],
        *TASK_TOPS[1:],
    ]
    recorder = answering(noisy)
    async with backend_for(recorder) as backend:
        decision = await backend.decide_task(TaskContext(message="hello"), READY)

    probe = backend.last_diagnostics["domain"]
    assert probe.observed_mass == pytest.approx(0.95)
    assert probe.valid_mass == pytest.approx(0.35)
    assert probe.invalid_mass == pytest.approx(0.60)
    assert probe.invalid_fraction == pytest.approx(0.60 / 0.95)
    assert probe.top_tokens[0] == ("Sure", pytest.approx(0.45))
    assert probe.label_mass["B"] == pytest.approx(0.30)

    # The renormalised numbers still only describe the valid mass.
    assert decision.domain_probabilities["coding"] == pytest.approx(0.30 / 0.35)
    assert sum(decision.domain_probabilities.values()) == pytest.approx(1.0)

    note = next(note for note in decision.notes if note.startswith("domain:"))
    assert "invalid" in note
    assert "'Sure'" in note


async def test_split_and_punctuated_labels_are_recorded_as_fallback_matches() -> None:
    tops: list[Tops] = [[("B)", 0.6), (" b", 0.2), ("zzz", 0.15)], *TASK_TOPS[1:]]
    recorder = answering(tops)
    async with backend_for(recorder) as backend:
        await backend.decide_task(TaskContext(message="hello"), READY)

    probe = backend.last_diagnostics["domain"]
    assert probe.label_mass["B"] == pytest.approx(0.8)
    assert probe.fallback_tokens == ("B)",)
    assert probe.invalid_mass == pytest.approx(0.15)


async def test_diagnostics_survive_as_structured_data_for_an_eval() -> None:
    recorder = answering(TASK_TOPS)
    async with backend_for(recorder) as backend:
        await backend.decide_task(TaskContext(message="hello"), READY)

    assert set(backend.last_diagnostics) == {"domain", "difficulty", "tool_need", "ambiguity"}
    described = backend.last_diagnostics["difficulty"].describe()
    assert described["key"] == "difficulty"
    assert described["top_tokens"][0] == ["2", 0.55]
    assert described["invalid_mass"] == pytest.approx(0.0, abs=1e-9)
    # Serialisable, because an eval will write these to disk.
    json.dumps(described)


async def test_no_valid_label_is_a_refusal_not_a_guess() -> None:
    tops: list[Tops] = [[("Sure", 0.6), ("I", 0.25), ("\n", 0.1)], *TASK_TOPS[1:]]
    recorder = answering(tops)
    async with backend_for(recorder) as backend:
        with pytest.raises(BackendError) as caught:
            await backend.decide_task(TaskContext(message="hello"), READY)

    assert caught.value.kind == "invalid"
    assert caught.value.retryable is False
    assert "domain" in str(caught.value)


# --- availability -----------------------------------------------------------


async def test_availability_is_unconfigured_while_the_experiment_is_off() -> None:
    recorder = answering(TASK_TOPS)
    async with backend_for(recorder, make_config(enabled=False)) as backend:
        availability = await backend.availability(READY)

    assert availability.state == "unconfigured"
    assert "decisions.logit.enabled" in availability.reason


async def test_availability_mirrors_the_local_backend_checks() -> None:
    recorder = answering(TASK_TOPS)
    async with backend_for(recorder) as backend:
        assert (await backend.availability(READY)).ready

        down = await backend.availability(BackendFacts(server_healthy=False))
        assert down.state == "unavailable"
        assert "/health" in down.reason

        busy = await backend.availability(
            BackendFacts(server_healthy=True, server_busy=True, resident_gen_tps=38.0)
        )
        assert busy.state == "unavailable"
        assert "--parallel 1" in busy.reason


async def test_a_slow_model_is_fine_by_default_but_can_be_fenced_off() -> None:
    """One token of output barely cares about tok/s, so the floor defaults to zero."""
    slow = BackendFacts(server_healthy=True, resident_model_id="27B.gguf", resident_gen_tps=3.0)
    recorder = answering(TASK_TOPS)
    async with backend_for(recorder) as backend:
        assert (await backend.availability(slow)).ready

    async with backend_for(recorder, make_config(min_resident_gen_tps=10.0)) as fenced:
        availability = await fenced.availability(slow)
    assert availability.state == "unavailable"
    assert "3.0 tok/s" in availability.reason


# --- failures ---------------------------------------------------------------


async def test_server_error_becomes_a_retryable_backend_error() -> None:
    recorder = failing(500, {"error": {"message": "failed to allocate slot"}})
    async with backend_for(recorder) as backend:
        with pytest.raises(BackendError) as caught:
            await backend.decide_task(TaskContext(message="hello"), READY)

    assert caught.value.backend == "logit"
    assert caught.value.kind == "server"
    assert caught.value.retryable is True


async def test_bad_request_becomes_a_non_retryable_backend_error() -> None:
    recorder = failing(400, {"error": {"message": "n_probs out of range"}})
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
            await backend.decide_step(StepContext(goal="g", action="a", result="r"), READY)

    assert caught.value.kind == "connection"


async def test_a_build_without_logprobs_is_reported_clearly() -> None:
    recorder = returning(
        {
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "B"}, "logprobs": None}
            ],
            "usage": {"prompt_tokens": 900, "completion_tokens": 1},
        }
    )
    async with backend_for(recorder) as backend:
        with pytest.raises(BackendError) as caught:
            await backend.decide_task(TaskContext(message="hello"), READY)

    assert caught.value.kind == "invalid"
    assert "logprobs" in str(caught.value)
