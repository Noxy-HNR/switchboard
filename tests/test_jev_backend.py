"""Tests for the Jev cloud decision backend.

Every case drives a `httpx2.MockTransport`, so nothing here touches the network
and no API key is needed. The handler keeps the raw `httpx2.Request`, which is
what lets the privacy tests assert on the exact bytes that would have left the
machine.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx2
import pytest
from typesafe_sdk import TypeSafeError

from switchboard.config import Config
from switchboard.decisions.base import BackendError, BackendFacts, StepContext, TaskContext
from switchboard.decisions.jev import JevBackend
from switchboard.decisions.schema import (
    AMBIGUITY_INSTRUCTIONS,
    DIFFICULTY_CRITERIA,
    DIFFICULTY_INSTRUCTIONS,
    DOMAIN_CRITERIA,
    DOMAIN_INSTRUCTIONS,
    DOMAINS,
    STEP_OUTCOME_CRITERIA,
    STEP_OUTCOME_INSTRUCTIONS,
    STEP_OUTCOMES,
    TOOL_NEED_CRITERIA,
    TOOL_NEED_INSTRUCTIONS,
    Difficulty,
)
from switchboard.privacy import prepare_cloud_text

FACTS = BackendFacts()
REQUEST_ID = "req_test_1"
HEADERS = {"x-typesafe-request-id": REQUEST_ID}

Handler = Callable[[httpx2.Request], httpx2.Response]


# --- fixtures and helpers ---------------------------------------------------


def make_config(
    *,
    api_key: str | None = "ts-unit-test-key",
    allow_cloud: bool = True,
    enabled: bool | str = "auto",
    max_retries: int = 0,
    max_state_chars: int = 1500,
    redact: bool = True,
) -> Config:
    """A Config built from defaults only, so no file or environment can sway a test."""
    config = Config()
    config.typesafe_api_key = api_key
    config.privacy.allow_cloud = allow_cloud
    config.privacy.max_state_chars = max_state_chars
    config.privacy.redact = redact
    config.decisions.jev.enabled = enabled
    config.decisions.jev.max_retries = max_retries
    return config


def recording(handler: Handler) -> tuple[httpx2.MockTransport, list[httpx2.Request]]:
    seen: list[httpx2.Request] = []

    def wrapper(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return handler(request)

    return httpx2.MockTransport(wrapper), seen


def responding(body: dict[str, Any], status: int = 200) -> tuple[httpx2.MockTransport, list[httpx2.Request]]:
    return recording(lambda _request: httpx2.Response(status, json=body, headers=HEADERS))


def exploding() -> httpx2.MockTransport:
    """A transport that fails the test if anything reaches it."""

    def handler(request: httpx2.Request) -> httpx2.Response:
        raise AssertionError(f"unexpected network call to {request.url}")

    return httpx2.MockTransport(handler)


def body_of(request: httpx2.Request) -> dict[str, Any]:
    return json.loads(request.content)


TASK_BODY: dict[str, Any] = {
    "model": "jev-2026-09-15",
    "usage": {"input_tokens": 500, "output_tokens": 12},
    "answers": {
        "domain": {
            "type": "choice",
            "choice": "coding",
            "confidence": 0.91,
            "probabilities": {"coding": 0.82, "general": 0.1, "system_ops": 0.08},
        },
        "difficulty": {
            "type": "score",
            # 1*0.2 + 2*0.3 + 3*0.5 == 2.3, matching the weighted score.
            "score": 2.3,
            "confidence": 0.77,
            "legend": {str(level): text for level, text in enumerate(DIFFICULTY_CRITERIA)},
            "probabilities": {"0": 0.0, "1": 0.2, "2": 0.3, "3": 0.5},
        },
        "tool_need": {"type": "noul", "noul": 0.93},
        "ambiguity": {"type": "noul", "noul": 0.07},
    },
}

STEP_BODY: dict[str, Any] = {
    "model": "jev-2026-09-15",
    "usage": {"input_tokens": 320, "output_tokens": 9},
    "answers": {
        "outcome": {
            "type": "choice",
            "choice": "failure",
            "confidence": 0.88,
            "probabilities": {"success": 0.03, "partial": 0.05, "failure": 0.88, "unclear": 0.04},
        },
        "progress": {"type": "noul", "noul": 0.12},
        "complete": {"type": "noul", "noul": 0.02},
        "needs_new_approach": {"type": "noul", "noul": 0.81},
    },
}


def task_context(message: str = "Why does my loader raise KeyError on the second pass?") -> TaskContext:
    return TaskContext(
        message=message,
        prior_message="It worked yesterday.",
        system_prompt="You are a helpful assistant. SYSTEMPROMPTCANARY",
        tool_names=["read_file", "run_command"],
        turn_index=3,
        estimated_prompt_tokens=412,
        has_images=True,
        session_id="abc123",
    )


def step_context() -> StepContext:
    return StepContext(
        goal="Get the test suite green",
        action="uv run pytest -q",
        result="2 failed, 18 passed",
        exit_code=1,
        attempt=2,
        previous_attempts=["ran pytest without installing deps"],
        elapsed_s=4.5,
        session_id="abc123",
    )


# --- task decisions ---------------------------------------------------------


async def test_decide_task_parses_a_full_answer() -> None:
    transport, seen = responding(TASK_BODY)
    backend = JevBackend(make_config(), transport=transport)

    decision = await backend.decide_task(task_context(), FACTS)

    assert len(seen) == 1, "all four questions must travel in one call"
    assert decision.backend == "jev"
    assert decision.domain == "coding"
    assert decision.domain_probabilities["coding"] == 0.82
    assert decision.domain_confidence == 0.91
    assert decision.difficulty == 2.3
    assert decision.difficulty_level is Difficulty.MODERATE
    assert decision.difficulty_probabilities == {"0": 0.0, "1": 0.2, "2": 0.3, "3": 0.5}
    assert decision.difficulty_confidence == 0.77
    assert decision.tool_need == 0.93
    assert decision.ambiguity == 0.07
    assert decision.degraded is False
    assert decision.latency_ms > 0.0
    # 500 input tokens at $0.042/Mtok == 21 micro-dollars; output tokens are free.
    assert decision.usage == {"input_tokens": 500, "output_tokens": 12, "est_cost_usd_micro": 21}
    assert decision.notes[0] == "model: jev-2026-09-15"
    assert f"request: {REQUEST_ID}" in decision.notes
    assert not any(note.startswith("redacted:") for note in decision.notes)

    await backend.aclose()


async def test_decide_task_sends_one_systemone_call_with_the_schema_questions() -> None:
    transport, seen = responding(TASK_BODY)
    backend = JevBackend(make_config(), transport=transport)

    await backend.decide_task(task_context(), FACTS)

    request = seen[0]
    assert request.method == "POST"
    assert str(request.url).endswith("/v1/systemone")
    body = body_of(request)
    assert body["model"] == "jev-latest"

    questions = body["questions"]
    assert set(questions) == {"domain", "difficulty", "tool_need", "ambiguity"}
    assert questions["domain"] == {
        "type": "choice",
        "instructions": DOMAIN_INSTRUCTIONS,
        "criteria": {name: DOMAIN_CRITERIA[name] for name in DOMAINS},
    }
    assert questions["difficulty"] == {
        "type": "score",
        "instructions": DIFFICULTY_INSTRUCTIONS,
        "criteria": list(DIFFICULTY_CRITERIA),
    }
    assert questions["tool_need"] == {
        "type": "noul",
        "instructions": TOOL_NEED_INSTRUCTIONS,
        "criteria": dict(TOOL_NEED_CRITERIA),
    }
    assert questions["ambiguity"]["instructions"] == AMBIGUITY_INSTRUCTIONS

    await backend.aclose()


async def test_task_state_carries_only_the_named_fields() -> None:
    transport, seen = responding(TASK_BODY)
    backend = JevBackend(make_config(), transport=transport)

    await backend.decide_task(task_context(), FACTS)

    # Only fields a question names (decisions.state). Tool names and the turn
    # number are facts the policy reads, not evidence any question judges.
    assert body_of(seen[0])["state"] == {
        "request": {
            "message": "Why does my loader raise KeyError on the second pass?",
            "previous_message": "It worked yesterday.",
            "has_images": True,
        }
    }
    # The system prompt is never sent: no question names it.
    assert b"SYSTEMPROMPTCANARY" not in (seen[0].content or b"")

    await backend.aclose()


async def test_task_state_omits_absent_context() -> None:
    transport, seen = responding(TASK_BODY)
    backend = JevBackend(make_config(), transport=transport)

    await backend.decide_task(TaskContext(message="hi"), FACTS)

    # An absent previous turn and a missing image are omitted, not sent as
    # empty values a literal reader might take as evidence.
    assert body_of(seen[0])["state"] == {"request": {"message": "hi"}}

    await backend.aclose()


async def test_long_message_goes_through_the_state_budget() -> None:
    transport, seen = responding(TASK_BODY)
    backend = JevBackend(make_config(max_state_chars=200), transport=transport)
    long_message = "a" * 5000

    await backend.decide_task(TaskContext(message=long_message), FACTS)

    # The contract is "prepare_cloud_text with privacy.max_state_chars", so the
    # test asserts against that helper rather than re-deriving its arithmetic.
    expected, _kinds = prepare_cloud_text(long_message, max_chars=200, do_redact=True)
    message = body_of(seen[0])["state"]["request"]["message"]
    assert message == expected
    assert len(message) < len(long_message) / 10

    await backend.aclose()


async def test_tool_names_are_not_sent() -> None:
    """A client's tool roster describes the client, not the request.

    Sending it made "does this need a tool?" lean toward yes for every request
    from a tool-heavy client. Whether tools are attached is a fact the policy
    reads directly (it excludes models that cannot express tool calls).
    """
    transport, seen = responding(TASK_BODY)
    backend = JevBackend(make_config(), transport=transport)

    context = TaskContext(message="do a thing", tool_names=[f"tool_{n}" for n in range(200)])
    await backend.decide_task(context, FACTS)

    sent = seen[0].content.decode()
    assert "tool_0" not in sent
    assert "available_tools" not in sent

    await backend.aclose()


# --- step decisions ---------------------------------------------------------


async def test_decide_step_parses_a_full_answer() -> None:
    transport, seen = responding(STEP_BODY)
    backend = JevBackend(make_config(), transport=transport)

    decision = await backend.decide_step(step_context(), FACTS)

    assert len(seen) == 1
    assert decision.backend == "jev"
    assert decision.outcome == "failure"
    assert decision.outcome_probabilities["failure"] == 0.88
    assert decision.outcome_confidence == 0.88
    assert decision.progress == 0.12
    assert decision.complete == 0.02
    assert decision.needs_new_approach == 0.81
    assert decision.degraded is False
    assert decision.usage == {"input_tokens": 320, "output_tokens": 9, "est_cost_usd_micro": 13}

    body = body_of(seen[0])
    assert set(body["questions"]) == {"outcome", "progress", "complete", "needs_new_approach"}
    assert body["questions"]["outcome"] == {
        "type": "choice",
        "instructions": STEP_OUTCOME_INSTRUCTIONS,
        "criteria": {name: STEP_OUTCOME_CRITERIA[name] for name in STEP_OUTCOMES},
    }
    assert body["state"] == {
        "task": {"goal": "Get the test suite green"},
        "step": {
            "action": "uv run pytest -q",
            "result": "2 failed, 18 passed",
            "exit_code": 1,
        },
        "earlier_attempts": ["ran pytest without installing deps"],
    }

    await backend.aclose()


async def test_step_state_omits_unknown_exit_code_and_empty_history() -> None:
    transport, seen = responding(STEP_BODY)
    backend = JevBackend(make_config(), transport=transport)

    await backend.decide_step(StepContext(goal="g", action="a", result="r"), FACTS)

    state = body_of(seen[0])["state"]
    assert "exit_code" not in state["step"]
    assert "earlier_attempts" not in state


# --- privacy ----------------------------------------------------------------


async def test_secrets_are_redacted_before_the_request_leaves() -> None:
    secret = "sk-proj-ABCD1234efgh5678IJKL"
    email = "dev@example.com"
    message = f"Fix the loader: it 401s with {secret} and then mail me at {email}"
    transport, seen = responding(TASK_BODY)
    backend = JevBackend(make_config(), transport=transport)

    decision = await backend.decide_task(TaskContext(message=message), FACTS)

    raw = (seen[0].content or b"").decode()
    assert secret not in raw
    assert email not in raw
    assert "[REDACTED:openai-key]" in raw
    assert "[REDACTED:email]" in raw
    # The surrounding text still gets through: redaction removes the secret, not the request.
    assert "Fix the loader" in raw
    assert "mail me at" in raw

    assert "redacted: openai-key, email" in decision.notes
    # The note names kinds, never values.
    assert all(secret not in note and email not in note for note in decision.notes)

    await backend.aclose()


async def test_redaction_can_be_switched_off_by_config() -> None:
    secret = "sk-proj-ABCD1234efgh5678IJKL"
    transport, seen = responding(TASK_BODY)
    backend = JevBackend(make_config(redact=False), transport=transport)

    decision = await backend.decide_task(TaskContext(message=f"key {secret}"), FACTS)

    assert secret in (seen[0].content or b"").decode()
    assert not any(note.startswith("redacted:") for note in decision.notes)

    await backend.aclose()


async def test_step_strings_are_redacted_too() -> None:
    secret = "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ012345"
    transport, seen = responding(STEP_BODY)
    backend = JevBackend(make_config(), transport=transport)

    await backend.decide_step(
        StepContext(goal="deploy", action="git push", result=f"denied for {secret}"), FACTS
    )

    assert secret not in (seen[0].content or b"").decode()


# --- availability -----------------------------------------------------------


async def test_unconfigured_without_an_api_key() -> None:
    backend = JevBackend(make_config(api_key=None), transport=exploding())

    availability = await backend.availability(FACTS)
    assert availability.state == "unconfigured"
    assert availability.ready is False
    assert "TYPESAFE_API_KEY" in availability.reason

    with pytest.raises(BackendError) as info:
        await backend.decide_task(task_context(), FACTS)
    assert info.value.kind == "skipped"
    assert info.value.retryable is False
    assert info.value.counts_against_health is False


async def test_unconfigured_when_cloud_is_not_allowed() -> None:
    backend = JevBackend(make_config(allow_cloud=False), transport=exploding())

    availability = await backend.availability(FACTS)
    assert availability.state == "unconfigured"
    assert "allow_cloud" in availability.reason

    with pytest.raises(BackendError) as info:
        await backend.decide_step(step_context(), FACTS)
    assert info.value.kind == "skipped"


async def test_unconfigured_when_disabled() -> None:
    backend = JevBackend(make_config(enabled=False), transport=exploding())

    availability = await backend.availability(FACTS)
    assert availability.state == "unconfigured"
    assert "enabled" in availability.reason


async def test_availability_is_ready_and_makes_no_network_call() -> None:
    transport, seen = responding(TASK_BODY)
    backend = JevBackend(make_config(), transport=transport)

    availability = await backend.availability(FACTS)

    assert availability.state == "ready"
    assert availability.ready is True
    assert seen == []


async def test_aclose_is_idempotent_and_stops_later_calls() -> None:
    transport, seen = responding(TASK_BODY)
    backend = JevBackend(make_config(), transport=transport)

    await backend.decide_task(task_context(), FACTS)
    await backend.aclose()
    await backend.aclose()

    assert (await backend.availability(FACTS)).state == "unconfigured"
    with pytest.raises(BackendError) as info:
        await backend.decide_task(task_context(), FACTS)
    assert info.value.kind == "skipped"
    assert len(seen) == 1


# --- error translation ------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "kind", "retryable", "counts"),
    [
        (401, "auth", False, False),
        (403, "auth", False, False),
        (429, "rate_limit", True, True),
        (422, "invalid", False, True),
        (400, "invalid", False, True),
        (500, "server", True, True),
        (503, "server", True, True),
    ],
)
async def test_http_errors_map_to_backend_error_kinds(
    status: int, kind: str, retryable: bool, counts: bool
) -> None:
    transport, seen = responding({"error": "nope"}, status=status)
    backend = JevBackend(make_config(), transport=transport)

    with pytest.raises(BackendError) as info:
        await backend.decide_task(task_context(), FACTS)

    assert info.value.kind == kind
    assert info.value.retryable is retryable
    assert info.value.counts_against_health is counts
    assert info.value.backend == "jev"
    assert len(seen) == 1, "max_retries=0 must not retry"


async def test_timeout_maps_to_timeout() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ReadTimeout("too slow", request=request)

    transport, _seen = recording(handler)
    backend = JevBackend(make_config(), transport=transport)

    with pytest.raises(BackendError) as info:
        await backend.decide_task(task_context(), FACTS)

    assert info.value.kind == "timeout"
    assert info.value.retryable is True
    assert info.value.counts_against_health is True


async def test_connection_failure_maps_to_connection() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("refused", request=request)

    transport, _seen = recording(handler)
    backend = JevBackend(make_config(), transport=transport)

    with pytest.raises(BackendError) as info:
        await backend.decide_step(step_context(), FACTS)

    assert info.value.kind == "connection"
    assert info.value.retryable is True


async def test_error_message_never_quotes_the_response_body() -> None:
    transport, _seen = responding({"error": "state was BODYCANARY"}, status=500)
    backend = JevBackend(make_config(), transport=transport)

    with pytest.raises(BackendError) as info:
        await backend.decide_task(task_context(), FACTS)

    assert "BODYCANARY" not in str(info.value)
    assert "status=500" in str(info.value)


async def test_max_retries_is_honoured() -> None:
    calls: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        calls.append(request)
        if len(calls) == 1:
            return httpx2.Response(503, json={"error": "try later"}, headers=HEADERS)
        return httpx2.Response(200, json=TASK_BODY, headers=HEADERS)

    backend = JevBackend(make_config(max_retries=1), transport=httpx2.MockTransport(handler))

    decision = await backend.decide_task(task_context(), FACTS)

    assert len(calls) == 2
    assert decision.domain == "coding"

    await backend.aclose()


# --- malformed answers ------------------------------------------------------


def _without(body: dict[str, Any], name: str) -> dict[str, Any]:
    answers = {key: value for key, value in body["answers"].items() if key != name}
    return {**body, "answers": answers}


@pytest.mark.parametrize("missing", ["domain", "difficulty", "tool_need", "ambiguity"])
async def test_a_missing_task_answer_is_invalid(missing: str) -> None:
    transport, _seen = responding(_without(TASK_BODY, missing))
    backend = JevBackend(make_config(), transport=transport)

    with pytest.raises(BackendError) as info:
        await backend.decide_task(task_context(), FACTS)

    assert info.value.kind == "invalid"
    assert info.value.retryable is False
    assert missing in str(info.value)


@pytest.mark.parametrize("missing", ["outcome", "progress", "complete", "needs_new_approach"])
async def test_a_missing_step_answer_is_invalid(missing: str) -> None:
    transport, _seen = responding(_without(STEP_BODY, missing))
    backend = JevBackend(make_config(), transport=transport)

    with pytest.raises(BackendError) as info:
        await backend.decide_step(step_context(), FACTS)

    assert info.value.kind == "invalid"
    assert missing in str(info.value)


async def test_an_unknown_domain_label_is_invalid() -> None:
    body = {
        **TASK_BODY,
        "answers": {
            **TASK_BODY["answers"],
            "domain": {
                "type": "choice",
                "choice": "banking",
                "confidence": 0.9,
                "probabilities": {"banking": 0.9, "coding": 0.1},
            },
        },
    }
    transport, _seen = responding(body)
    backend = JevBackend(make_config(), transport=transport)

    with pytest.raises(BackendError) as info:
        await backend.decide_task(task_context(), FACTS)

    assert info.value.kind == "invalid"
    assert "banking" in str(info.value)


async def test_an_out_of_range_difficulty_is_invalid() -> None:
    body = {
        **TASK_BODY,
        "answers": {
            **TASK_BODY["answers"],
            "difficulty": {**TASK_BODY["answers"]["difficulty"], "score": 7.5},
        },
    }
    transport, _seen = responding(body)
    backend = JevBackend(make_config(), transport=transport)

    with pytest.raises(BackendError) as info:
        await backend.decide_task(task_context(), FACTS)

    assert info.value.kind == "invalid"


async def test_a_wrong_answer_type_is_invalid() -> None:
    body = {
        **TASK_BODY,
        "answers": {**TASK_BODY["answers"], "tool_need": TASK_BODY["answers"]["domain"]},
    }
    transport, _seen = responding(body)
    backend = JevBackend(make_config(), transport=transport)

    with pytest.raises(BackendError) as info:
        await backend.decide_task(task_context(), FACTS)

    assert info.value.kind == "invalid"
    assert "wrong type" in str(info.value)


async def test_an_unparseable_body_is_invalid() -> None:
    body = {**TASK_BODY, "usage": {"input_tokens": "many", "output_tokens": 12}}
    transport, _seen = responding(body)
    backend = JevBackend(make_config(), transport=transport)

    with pytest.raises(BackendError) as info:
        await backend.decide_task(task_context(), FACTS)

    assert info.value.kind == "invalid"
    assert info.value.retryable is False


async def test_usage_keys_are_omitted_when_the_api_reports_none() -> None:
    body = {**TASK_BODY, "usage": {}}
    transport, _seen = responding(body)
    backend = JevBackend(make_config(), transport=transport)

    decision = await backend.decide_task(task_context(), FACTS)

    assert decision.usage == {}


# --- construction -----------------------------------------------------------


def test_client_and_transport_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError):
        JevBackend(make_config(), client=object(), transport=exploding())  # type: ignore[arg-type]


async def test_a_bad_timeout_is_reported_as_invalid_not_a_crash() -> None:
    config = make_config()
    config.decisions.jev.timeout_s = 0.0
    backend = JevBackend(config, transport=exploding())

    with pytest.raises(BackendError) as info:
        await backend.decide_task(task_context(), FACTS)

    assert info.value.kind == "invalid"
    assert isinstance(info.value.__cause__, TypeSafeError)


# --- review follow-ups ------------------------------------------------------


async def test_both_decisions_are_marked_calibrated() -> None:
    """Jev is the one backend trained for calibrated probabilities; the policy
    weighs a distribution only when the decision says it may."""
    transport, _seen = recording(
        lambda request: httpx2.Response(
            200,
            json=STEP_BODY if "outcome" in body_of(request)["questions"] else TASK_BODY,
            headers=HEADERS,
        )
    )
    backend = JevBackend(make_config(), transport=transport)

    task = await backend.decide_task(task_context(), FACTS)
    step = await backend.decide_step(step_context(), FACTS)

    assert task.calibrated is True
    assert step.calibrated is True
    assert task.summary()["calibrated"] is True

    await backend.aclose()


async def test_a_step_whose_result_is_machine_output_is_judged_locally() -> None:
    """Minimization would leave the step questions judging a placeholder, so the
    step is declined (`skipped`, not an outage) before anything is sent."""
    backend = JevBackend(make_config(), transport=exploding())
    context = StepContext(
        goal="load the fixtures",
        action="python load.py",
        result=(
            "Traceback (most recent call last):\n"
            '  File "load.py", line 3, in <module>\n'
            "    rows = read_rows(path)\n"
            "KeyError: 'id'\n"
        ),
        exit_code=1,
    )

    with pytest.raises(BackendError) as info:
        await backend.decide_step(context, FACTS)

    assert info.value.kind == "skipped"
    assert info.value.retryable is False
    assert "judged locally" in str(info.value)

    await backend.aclose()


@pytest.mark.parametrize("score,expected", [(3.0000004, 3.0), (-0.0000004, 0.0)])
async def test_a_score_a_hair_outside_its_range_is_clamped(score: float, expected: float) -> None:
    """Float round-off in a probability-weighted mean is not an invalid answer."""
    body = {
        **TASK_BODY,
        "answers": {
            **TASK_BODY["answers"],
            "difficulty": {**TASK_BODY["answers"]["difficulty"], "score": score},
        },
    }
    transport, _seen = responding(body)
    backend = JevBackend(make_config(), transport=transport)

    decision = await backend.decide_task(task_context(), FACTS)

    assert decision.difficulty == expected

    await backend.aclose()


async def test_a_noul_a_hair_above_one_is_clamped() -> None:
    body = {
        **TASK_BODY,
        "answers": {**TASK_BODY["answers"], "tool_need": {"type": "noul", "noul": 1.0000004}},
    }
    transport, _seen = responding(body)
    backend = JevBackend(make_config(), transport=transport)

    decision = await backend.decide_task(task_context(), FACTS)

    assert decision.tool_need == 1.0

    await backend.aclose()


def test_the_sdk_debug_logger_is_raised_to_info() -> None:
    """The SDK's debug level logs request bodies unredacted; building the backend
    must not leave that on, whatever TYPESAFE_LOG_LEVEL said."""
    import logging

    sdk_logger = logging.getLogger("typesafe_sdk")
    before = sdk_logger.level
    sdk_logger.setLevel(logging.DEBUG)
    try:
        JevBackend(make_config(), transport=exploding())
        assert sdk_logger.getEffectiveLevel() >= logging.INFO
    finally:
        sdk_logger.setLevel(before)
