"""Tests for the deterministic backend.

The rules backend is the chain's floor, so the tests are written against the
two things that actually matter when it runs: that it routes a real request to
a defensible domain, and that it never lies about how sure it is. The domain
table is the bulk of it - one row per representative request, covering all six
domains - because a keyword engine fails by drifting, and a table is the only
thing that catches drift.

Nothing here asserts an exact score. Scores are implementation; the domain
chosen, the ordering of difficulties, the discrete level of a probability and
the invariants (sums, bounds, purity) are the contract.
"""

from __future__ import annotations

import time

import pytest

from switchboard.decisions.base import (
    BackendFacts,
    DecisionBackend,
    StepContext,
    TaskContext,
)
from switchboard.decisions.rules import (
    HEURISTIC_NOTE,
    RulesBackend,
    step_signals,
    task_signals,
)
from switchboard.decisions.schema import DOMAINS, STEP_OUTCOMES, score_from_probabilities

FACTS = BackendFacts()

TRACEBACK = (
    "Traceback (most recent call last):\n"
    '  File "C:\\AI\\Switchboard\\run.py", line 42, in <module>\n'
    "    main()\n"
    "ValueError: not enough values to unpack"
)


@pytest.fixture
def backend() -> RulesBackend:
    return RulesBackend()


async def task(backend: RulesBackend, message: str, **kwargs: object):
    return await backend.decide_task(TaskContext(message=message, **kwargs), FACTS)  # type: ignore[arg-type]


async def step(backend: RulesBackend, **kwargs: object):
    context = StepContext(
        goal=str(kwargs.pop("goal", "make the build pass")),
        action=str(kwargs.pop("action", "uv run pytest -q")),
        result=str(kwargs.pop("result", "")),
        **kwargs,  # type: ignore[arg-type]
    )
    return await backend.decide_step(context, FACTS)


# --- Domain routing ---------------------------------------------------------

# One row per representative request. Every domain in DOMAINS appears, and the
# awkward cases are deliberate: a medical question that mentions numbers, a
# Windows terminal request, a coding request that is mostly a stack trace, a
# greeting, and a one-liner with no content at all.
DOMAIN_CASES: list[tuple[str, str]] = [
    ("hey there, good morning!", "general"),
    ("Thanks, that's exactly what I needed.", "general"),
    ("What's the capital of Australia?", "general"),
    ("Explain how a refrigerator works in simple terms.", "general"),
    ("fix it", "general"),
    ("Why does this Python function raise a KeyError when the dict is empty?", "coding"),
    (TRACEBACK, "coding"),
    ("```python\nfor i in range(10): print(i)\n```\nMake this faster.", "coding"),
    ("Review this pull request and tell me if the error handling is sound.", "coding"),
    ("Write a pytest fixture that spins up a temporary SQLite database.", "coding"),
    (
        "My father is 68 and his blood pressure is 160/95 - should he start an ACE inhibitor?",
        "medical",
    ),
    ("What are the common side effects of metformin?", "medical"),
    ("I've had a fever and a rash for three days. Should I see a doctor?", "medical"),
    ("Summarise my CT scan report: 4 mm nodule in the right upper lobe.", "medical"),
    ("Draft a polite email telling the client we need two more weeks.", "writing"),
    ("Rewrite this paragraph so it sounds less formal.", "writing"),
    ("Proofread my cover letter for typos and awkward phrasing.", "writing"),
    (
        "Compare the pros and cons of Postgres versus SQLite for a desktop app, with sources.",
        "research",
    ),
    (
        "What does the current literature say about intermittent fasting "
        "and weight loss? Cite studies.",
        "research",
    ),
    (
        "Compare three laptops under $1200 on battery life and Linux support, "
        "and list the trade-offs.",
        "research",
    ),
    ("How do I see which process is using port 10000 on Windows?", "system_ops"),
    ("Run `Get-Service | Where-Object Status -eq Running` and show me the output.", "system_ops"),
    ("Delete the folder C:\\AI\\models\\old and free up disk space.", "system_ops"),
    (
        "Docker Desktop starts on boot and eats 4 GB of RAM - how do I stop it from starting?",
        "system_ops",
    ),
    ("Install ffmpeg with winget and add it to PATH.", "system_ops"),
]


def test_domain_table_covers_every_domain() -> None:
    assert {expected for _, expected in DOMAIN_CASES} == set(DOMAINS)
    assert len(DOMAIN_CASES) >= 20


@pytest.mark.parametrize(
    ("message", "expected"),
    DOMAIN_CASES,
    ids=[f"{expected}-{index}" for index, (_, expected) in enumerate(DOMAIN_CASES)],
)
async def test_domain_routing(backend: RulesBackend, message: str, expected: str) -> None:
    decision = await task(backend, message)
    assert decision.domain == expected, decision.notes


async def test_attached_tools_are_evidence_not_a_verdict(backend: RulesBackend) -> None:
    """Tools nudge a tie; they never drag a clear medical question to system_ops."""
    message = "What are the common side effects of metformin?"
    tools = ["run_powershell", "read_file", "list_processes", "take_screenshot"]

    with_tools = await task(backend, message, tool_names=tools)
    assert with_tools.domain == "medical"

    # ... but on a request with no lexical cues at all, they do move the needle.
    neutral = await task(backend, "have a look at this", tool_names=tools)
    assert neutral.domain_probabilities["system_ops"] > neutral.domain_probabilities["writing"]


# --- Difficulty -------------------------------------------------------------


async def test_difficulty_is_ordered(backend: RulesBackend) -> None:
    greeting = await task(backend, "hey, good morning!")
    routine = await task(backend, "Write a Python function that reverses a string.")
    multipart = await task(
        backend,
        "Design a caching layer for our API, explain the trade-offs between LRU "
        "and LFU, and then write the Python implementation with tests.",
    )

    assert greeting.difficulty < routine.difficulty < multipart.difficulty
    assert greeting.difficulty_level == 0
    assert multipart.difficulty_level >= 2


async def test_difficulty_rises_with_a_real_error_in_hand(backend: RulesBackend) -> None:
    plain = await task(backend, "How do I read a file in Python?")
    with_trace = await task(backend, f"My script blew up, what is wrong?\n{TRACEBACK}")
    assert with_trace.difficulty > plain.difficulty


async def test_difficulty_is_clamped_to_the_scale(backend: RulesBackend) -> None:
    piled_on = (
        "Design and prove the architecture, "
        + "compare the trade-offs and then optimise the concurrency, " * 40
    )
    decision = await task(backend, piled_on)
    assert 0.0 <= decision.difficulty <= 3.0

    decision = await task(backend, "ok")
    assert 0.0 <= decision.difficulty <= 3.0


@pytest.mark.parametrize(
    "message",
    [
        "hi",
        "hey there, good morning!",
        "Thanks, that's exactly what I needed.",
        "ok",
    ],
)
async def test_small_talk_is_trivial(backend: RulesBackend, message: str) -> None:
    decision = await task(backend, message)
    assert decision.difficulty_level == 0


async def test_difficulty_probabilities_reproduce_the_score(backend: RulesBackend) -> None:
    """The distribution is a presentation of the float, not a second opinion."""
    for message, _ in DOMAIN_CASES:
        decision = await task(backend, message)
        recovered = score_from_probabilities(decision.difficulty_probabilities, 4)
        assert recovered == pytest.approx(decision.difficulty, abs=1e-9)
        assert sum(decision.difficulty_probabilities.values()) == pytest.approx(1.0)


# --- tool_need and ambiguity ------------------------------------------------


@pytest.mark.parametrize(
    ("message", "tools", "expected"),
    [
        ("What's the difference between a list and a tuple in Python?", [], 0.1),
        ("Rewrite this paragraph to be shorter.", [], 0.1),
        ("Why is the sky blue?", ["read_file", "run_command"], 0.1),
        # Asking *about* a command is not running it.
        ("How do I list the running processes in PowerShell?", [], 0.35),
        ("Find the largest files in my downloads folder", [], 0.6),
        ("Run `pytest -q` in C:\\AI\\Switchboard and tell me which tests fail.", [], 0.85),
        ("Open the folder C:\\AI\\models", [], 0.85),
        ("Open the log file and tell me what the last error was.", ["read_file"], 0.85),
    ],
)
async def test_tool_need_levels(
    backend: RulesBackend, message: str, tools: list[str], expected: float
) -> None:
    decision = await task(backend, message, tool_names=tools)
    assert decision.tool_need == expected, decision.notes


async def test_tool_need_uses_only_discrete_levels(backend: RulesBackend) -> None:
    levels = {0.1, 0.35, 0.6, 0.85}
    for message, _ in DOMAIN_CASES:
        decision = await task(backend, message)
        assert decision.tool_need in levels
        assert decision.ambiguity in levels


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("fix it", 0.85),
        ("make it better", 0.85),
        ("What's the capital of Australia?", 0.1),
        ("Write a pytest fixture that spins up a temporary SQLite database.", 0.1),
        ("hi", 0.1),
    ],
)
async def test_ambiguity_levels(backend: RulesBackend, message: str, expected: float) -> None:
    decision = await task(backend, message)
    assert decision.ambiguity == expected, decision.notes


async def test_a_prior_turn_supplies_the_missing_referent(backend: RulesBackend) -> None:
    alone = await task(backend, "fix it")
    in_context = await task(backend, "fix it", prior_message="the build failed on line 3")
    assert in_context.ambiguity < alone.ambiguity


# --- Honesty ----------------------------------------------------------------


async def test_every_decision_says_it_is_heuristic(backend: RulesBackend) -> None:
    decision = await task(backend, "Write a Python function that reverses a string.")
    assert any("heuristic" in note for note in decision.notes)
    assert HEURISTIC_NOTE in decision.notes

    evaluation = await step(backend, result="Successfully installed ffmpeg", exit_code=0)
    assert any("heuristic" in note for note in evaluation.notes)


async def test_domain_probabilities_are_a_normalised_score_vector(backend: RulesBackend) -> None:
    for message, expected in DOMAIN_CASES:
        decision = await task(backend, message)
        probabilities = decision.domain_probabilities
        assert set(probabilities) == set(DOMAINS)
        assert sum(probabilities.values()) == pytest.approx(1.0)
        assert all(value > 0.0 for value in probabilities.values())
        assert max(probabilities, key=lambda key: probabilities[key]) == expected


async def test_confidence_never_claims_certainty(backend: RulesBackend) -> None:
    for message, _ in DOMAIN_CASES:
        decision = await task(backend, message)
        assert decision.domain_confidence is not None
        assert 0.2 <= decision.domain_confidence < 0.75
        assert decision.difficulty_confidence is not None
        assert 0.25 <= decision.difficulty_confidence <= 0.6


async def test_confidence_tracks_the_margin(backend: RulesBackend) -> None:
    """A request that only one domain claims beats one that two domains split."""
    clear = await task(
        backend,
        "Proofread my cover letter: fix the grammar, the punctuation and the wording.",
    )
    contested = await task(
        backend,
        "My doctor wants a Python script that charts my blood pressure readings.",
    )
    assert clear.domain_confidence > contested.domain_confidence


async def test_notes_never_echo_the_request(backend: RulesBackend) -> None:
    """Notes are storable: they carry cue labels and counts, never prompt text."""
    secret = "sk-liveKEY0000000000 my mother's maiden name is Vandermeer"
    decision = await task(backend, f"Deploy the service now. {secret}")
    joined = " ".join(decision.notes)
    assert "Vandermeer" not in joined
    assert "sk-live" not in joined
    assert "maiden" not in joined


# --- Step judgments ---------------------------------------------------------

STEP_CASES: list[tuple[str, int | None, str]] = [
    ("npm ERR! code ENOENT", 1, "failure"),
    # A hard fact beats a friendly-looking result.
    ("Done. Everything completed successfully.", 3, "failure"),
    (TRACEBACK, None, "failure"),
    ("error: cannot find module 'switchboard'", None, "failure"),
    ("The system cannot find the file specified.", None, "failure"),
    ("Permission denied", None, "failure"),
    ("The command failed with exit code 9", None, "failure"),
    ("", 0, "unclear"),
    ("", None, "unclear"),
    ("   \n   ", None, "unclear"),
    ("models\nConfig\nstart.ps1", None, "unclear"),
    ("4 passed, 0 failed in 0.31s", 0, "success"),
    ("Successfully installed ffmpeg", 0, "success"),
    ("models\nConfig\nstart.ps1", 0, "success"),
    ("Wrote 3 files; 1 file failed to write: permission denied", None, "partial"),
]


@pytest.mark.parametrize(
    ("result", "exit_code", "expected"),
    STEP_CASES,
    ids=[f"{expected}-{index}" for index, (_, _, expected) in enumerate(STEP_CASES)],
)
async def test_step_outcomes(
    backend: RulesBackend, result: str, exit_code: int | None, expected: str
) -> None:
    decision = await step(backend, result=result, exit_code=exit_code)
    assert decision.outcome == expected, decision.notes


async def test_a_zero_count_is_not_an_error(backend: RulesBackend) -> None:
    """'0 failed' must not read as 'failed'."""
    signals = step_signals(
        StepContext(goal="g", action="pytest", result="12 passed, 0 failed, 0 errors")
    )
    assert signals.errors == []
    assert signals.successes


async def test_repeated_attempt_demands_a_new_approach(backend: RulesBackend) -> None:
    first = await step(
        backend,
        action="uv run pytest -q",
        result="ERROR: file not found",
        exit_code=2,
    )
    again = await step(
        backend,
        action="uv run pytest -q",
        result="ERROR: file not found",
        exit_code=2,
        attempt=2,
        previous_attempts=["uv run  pytest -q"],  # whitespace differences still count as identical
    )
    assert again.needs_new_approach > first.needs_new_approach
    assert again.needs_new_approach >= 0.8
    assert again.progress <= 0.15


async def test_inherent_and_transient_failures_are_told_apart(backend: RulesBackend) -> None:
    inherent = await step(backend, result="Permission denied", exit_code=1)
    transient = await step(backend, result="Connection timed out after 30s", exit_code=1)
    assert inherent.needs_new_approach >= 0.8
    assert transient.needs_new_approach <= 0.35
    assert transient.needs_new_approach < inherent.needs_new_approach


async def test_repeated_failures_wear_down_the_transient_excuse(backend: RulesBackend) -> None:
    once = await step(backend, result="Connection timed out", exit_code=1, attempt=1)
    thrice = await step(backend, result="Connection timed out", exit_code=1, attempt=3)
    assert thrice.needs_new_approach > once.needs_new_approach


async def test_success_reports_progress_without_claiming_the_task_is_done(
    backend: RulesBackend,
) -> None:
    decision = await step(
        backend,
        goal="install ffmpeg and add it to PATH",
        action="winget install ffmpeg",
        result="Successfully installed ffmpeg",
        exit_code=0,
    )
    assert decision.outcome == "success"
    assert decision.progress >= 0.7
    assert decision.complete <= 0.5


async def test_complete_stays_conservative_everywhere(backend: RulesBackend) -> None:
    for result, exit_code, _ in STEP_CASES:
        decision = await step(backend, result=result, exit_code=exit_code)
        assert decision.complete <= 0.5


async def test_outcome_probabilities_are_normalised(backend: RulesBackend) -> None:
    for result, exit_code, expected in STEP_CASES:
        decision = await step(backend, result=result, exit_code=exit_code)
        probabilities = decision.outcome_probabilities
        assert set(probabilities) == set(STEP_OUTCOMES)
        assert sum(probabilities.values()) == pytest.approx(1.0)
        assert max(probabilities, key=lambda key: probabilities[key]) == expected
        assert decision.outcome_confidence is not None
        assert 0.2 <= decision.outcome_confidence < 0.75


# --- Contract ---------------------------------------------------------------


async def test_it_implements_the_backend_protocol(backend: RulesBackend) -> None:
    assert isinstance(backend, DecisionBackend)
    assert backend.name == "rules"


@pytest.mark.parametrize(
    "facts",
    [
        BackendFacts(),
        BackendFacts(server_healthy=False, server_busy=True),
        BackendFacts(resident_model_id="qwen", resident_gen_tps=0.4, server_busy=True),
    ],
)
async def test_availability_is_always_ready(backend: RulesBackend, facts: BackendFacts) -> None:
    availability = await backend.availability(facts)
    assert availability.ready
    assert availability.state == "ready"


async def test_decisions_are_labelled_and_timed(backend: RulesBackend) -> None:
    decision = await task(backend, "Install ffmpeg with winget.")
    assert decision.backend == "rules"
    assert decision.degraded is False
    assert decision.latency_ms >= 0.0
    assert decision.usage == {}

    evaluation = await step(backend, result="ok", exit_code=0)
    assert evaluation.backend == "rules"
    assert evaluation.degraded is False


@pytest.mark.parametrize(
    "message",
    [
        "",
        "   ",
        "?" * 5000,
        "\x00\ufeff \U0001f600 ??? ```",
        "C:\\" * 500,
        "1234567890 " * 400,
    ],
)
async def test_pathological_input_still_decides(backend: RulesBackend, message: str) -> None:
    decision = await task(backend, message, tool_names=[f"tool_{index}" for index in range(50)])
    assert decision.domain in DOMAINS
    assert 0.0 <= decision.difficulty <= 3.0
    assert 0.0 <= decision.tool_need <= 1.0
    assert 0.0 <= decision.ambiguity <= 1.0
    assert decision.degraded is False


async def test_it_is_pure(backend: RulesBackend) -> None:
    context = TaskContext(
        message="Delete the folder C:\\AI\\models\\old and free up disk space.",
        tool_names=["run_powershell"],
    )
    before = (context.message, list(context.tool_names), context.turn_index)

    first = await backend.decide_task(context, FACTS)
    second = await backend.decide_task(context, FACTS)

    assert (context.message, list(context.tool_names), context.turn_index) == before
    assert first.summary() | {"latency_ms": 0} == second.summary() | {"latency_ms": 0}


async def test_it_is_fast_enough_to_be_a_fallback(backend: RulesBackend) -> None:
    """The point of this backend is that it costs nothing, so measure it."""
    context = TaskContext(message=DOMAIN_CASES[5][0] + " " + TRACEBACK)
    start = time.perf_counter()
    for _ in range(200):
        await backend.decide_task(context, FACTS)
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0, f"200 decisions took {elapsed:.3f}s"


async def test_aclose_is_a_no_op(backend: RulesBackend) -> None:
    await backend.aclose()
    decision = await task(backend, "still works afterwards?")
    assert decision.domain in DOMAINS


async def test_signals_are_inspectable() -> None:
    """Every point scored has a name attached; that is the whole contract."""
    signals = task_signals(
        TaskContext(
            message="Run `Get-Service` and paste the output into C:\\AI\\report.log",
            tool_names=["run_powershell"],
        )
    )
    assert "powershell-cmdlet" in signals.structural
    assert "windows-path" in signals.structural
    assert "ops-file" in signals.structural
    assert signals.tool_hits.get("system_ops")
    assert signals.top_domains(1)[0][0] == "system_ops"
