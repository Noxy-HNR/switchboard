"""The decision chain: fallback, budget, breaker, cache and the fast path.

The chain is what keeps a decision provider from becoming a dependency. Each
test here is one way that could go wrong: a provider that refuses, one that
fails, one that hangs, and one that is simply not worth calling.
"""

from __future__ import annotations

import asyncio

import pytest

from switchboard.config import Config
from switchboard.decisions.base import (
    Availability,
    BackendError,
    BackendFacts,
    StepContext,
    TaskContext,
)
from switchboard.decisions.chain import DecisionChain
from switchboard.decisions.rules import RulesBackend
from switchboard.decisions.schema import StepDecision, TaskDecision

FACTS = BackendFacts(
    resident_model_id="fast-9b", resident_gen_tps=30.0, server_healthy=True, server_busy=False
)

EASY = TaskContext(message="hi there", estimated_prompt_tokens=3)
HARD = TaskContext(
    message=(
        "Design a scheduling policy for a single-GPU inference server that must serve "
        "interactive chat and batch jobs without starving either, and justify the trade-offs."
    ),
    estimated_prompt_tokens=40,
)
MEDICAL = TaskContext(
    message="Is it safe to take ibuprofen with lisinopril, and what should I watch for?",
    estimated_prompt_tokens=20,
)


class Recorder:
    """A backend that records that it was called, and answers instantly."""

    def __init__(self, name: str = "recorder", *, domain: str = "coding") -> None:
        self.name = name
        self.calls = 0
        self._domain = domain

    async def availability(self, facts: BackendFacts) -> Availability:
        return Availability.ok()

    async def decide_task(self, context: TaskContext, facts: BackendFacts) -> TaskDecision:
        self.calls += 1
        return TaskDecision(domain=self._domain, difficulty=2.5, backend=self.name)

    async def decide_step(self, context: StepContext, facts: BackendFacts) -> StepDecision:
        self.calls += 1
        return StepDecision(outcome="success", backend=self.name)

    async def aclose(self) -> None:
        return None


class Failing(Recorder):
    def __init__(self, kind: str = "server") -> None:
        super().__init__("failing")
        self.kind = kind

    async def decide_task(self, context: TaskContext, facts: BackendFacts) -> TaskDecision:
        self.calls += 1
        raise BackendError("nope", backend=self.name, kind=self.kind)  # type: ignore[arg-type]


class Refusing(Recorder):
    def __init__(self) -> None:
        super().__init__("refusing")

    async def availability(self, facts: BackendFacts) -> Availability:
        return Availability.unconfigured("no credential")


class Slow(Recorder):
    def __init__(self, delay: float = 5.0) -> None:
        super().__init__("slow")
        self.delay = delay

    async def decide_task(self, context: TaskContext, facts: BackendFacts) -> TaskDecision:
        self.calls += 1
        await asyncio.sleep(self.delay)
        return TaskDecision(backend=self.name)


def chain_of(config: Config, *backends) -> DecisionChain:
    return DecisionChain(config, list(backends))


# --- fallback ---------------------------------------------------------------


async def test_the_first_backend_that_answers_wins(config: Config) -> None:
    config.decisions.fast_path.enabled = False
    first, second = Recorder("first"), Recorder("second")
    result = await chain_of(config, first, second).decide_task(HARD, FACTS)

    assert result.decision.backend == "first"
    assert second.calls == 0


async def test_a_refusal_is_not_a_failure(config: Config) -> None:
    config.decisions.fast_path.enabled = False
    refusing, backup = Refusing(), Recorder("backup")
    chain = chain_of(config, refusing, backup)
    result = await chain.decide_task(HARD, FACTS)

    assert result.decision.backend == "backup"
    assert result.fell_back is True
    attempt = result.attempts[0]
    assert attempt.error_kind == "unconfigured"
    assert attempt.skipped_reason == "no credential"
    # A refusal must not count against the breaker.
    assert chain.health()["refusing"]["consecutive_failures"] == 0


async def test_a_failure_falls_through_and_counts(config: Config) -> None:
    config.decisions.fast_path.enabled = False
    failing, backup = Failing(), Recorder("backup")
    chain = chain_of(config, failing, backup)
    result = await chain.decide_task(HARD, FACTS)

    assert result.decision.backend == "backup"
    assert chain.health()["failing"]["consecutive_failures"] == 1


async def test_the_breaker_stops_calling_a_broken_backend(config: Config) -> None:
    config.decisions.fast_path.enabled = False
    config.decisions.jev.breaker_failures = 2
    failing, backup = Failing(), Recorder("backup")
    chain = chain_of(config, failing, backup)

    for index in range(4):
        # Vary the text so the cache does not answer instead.
        await chain.decide_task(
            TaskContext(message=f"a hard question number {index}"), FACTS, use_cache=False
        )

    assert failing.calls == 2, "the breaker should have opened after two failures"
    assert chain.health()["failing"]["open"] is True


async def test_an_auth_failure_does_not_open_the_breaker(config: Config) -> None:
    """A missing credential is a configuration problem, not an outage."""
    config.decisions.fast_path.enabled = False
    failing, backup = Failing(kind="auth"), Recorder("backup")
    chain = chain_of(config, failing, backup)
    await chain.decide_task(HARD, FACTS)

    assert chain.health()["failing"]["consecutive_failures"] == 0


async def test_the_budget_stops_a_slow_backend(config: Config) -> None:
    config.decisions.fast_path.enabled = False
    config.decisions.budget_ms = 200
    slow, backup = Slow(delay=5.0), Recorder("backup")
    result = await chain_of(config, slow, backup).decide_task(HARD, FACTS)

    assert result.decision.backend == "backup"
    assert result.attempts[0].error_kind == "timeout"


async def test_the_floor_runs_even_with_no_budget_left(config: Config) -> None:
    """A spent budget must still produce a decision."""
    config.decisions.fast_path.enabled = False
    config.decisions.budget_ms = 0
    slow = Slow(delay=5.0)
    result = await chain_of(config, slow, RulesBackend(config)).decide_task(HARD, FACTS)

    assert result.decision.backend == "rules"


async def test_an_empty_chain_still_returns_a_decision(config: Config) -> None:
    result = await chain_of(config).decide_task(HARD, FACTS)

    assert result.decision.backend == "none"
    assert result.decision.degraded is True
    assert result.decision.notes


# --- cache ------------------------------------------------------------------


async def test_the_same_question_is_not_asked_twice(config: Config) -> None:
    config.decisions.fast_path.enabled = False
    backend = Recorder("once")
    chain = chain_of(config, backend)

    first = await chain.decide_task(HARD, FACTS)
    second = await chain.decide_task(HARD, FACTS)

    assert backend.calls == 1
    assert second.cached is True
    # Equal, not identical: the cache hands out copies so one caller's notes
    # cannot leak into another's result.
    assert first.decision == second.decision
    assert first.decision is not second.decision


# --- the fast path ----------------------------------------------------------


async def test_an_easy_request_never_reaches_the_expensive_backend(config: Config) -> None:
    expensive = Recorder("expensive")
    chain = chain_of(config, expensive, RulesBackend(config))
    result = await chain.decide_task(EASY, FACTS)

    assert expensive.calls == 0, "an easy request must not pay for a semantic backend"
    assert result.decision.backend == "rules"
    assert any("fast path" in note for note in result.decision.notes)
    assert result.decision.latency_ms < 50


async def test_a_hard_request_does_reach_it(config: Config) -> None:
    expensive = Recorder("expensive")
    chain = chain_of(config, expensive, RulesBackend(config))
    result = await chain.decide_task(HARD, FACTS)

    assert expensive.calls == 1
    assert result.decision.backend == "expensive"


async def test_medical_always_gets_the_full_chain(config: Config) -> None:
    """A wrong domain here means the specialist model is never reached."""
    expensive = Recorder("expensive", domain="medical")
    chain = chain_of(config, expensive, RulesBackend(config))
    result = await chain.decide_task(MEDICAL, FACTS)

    assert expensive.calls == 1
    assert result.decision.backend == "expensive"


async def test_the_fast_path_can_be_turned_off(config: Config) -> None:
    config.decisions.fast_path.enabled = False
    expensive = Recorder("expensive")
    chain = chain_of(config, expensive, RulesBackend(config))
    await chain.decide_task(EASY, FACTS)

    assert expensive.calls == 1


async def test_the_fast_path_does_not_apply_to_steps(config: Config) -> None:
    expensive = Recorder("expensive")
    chain = chain_of(config, expensive, RulesBackend(config))
    result = await chain.decide_step(
        StepContext(goal="g", action="a", result="done", exit_code=0), FACTS
    )

    assert expensive.calls == 1
    assert result.decision.backend == "expensive"


# --- health -----------------------------------------------------------------


async def test_probe_reports_every_backend(config: Config) -> None:
    chain = chain_of(config, Refusing(), RulesBackend(config))
    health = await chain.probe(FACTS)

    assert health["refusing"]["availability"] == "unconfigured"
    assert health["rules"]["availability"] == "ready"


async def test_a_meaningful_medical_share_is_confirmed_by_the_full_chain(config: Config) -> None:
    """A request the rules rate mostly general but partly medical still gets checked."""
    expensive = Recorder("expensive", domain="medical")

    class MostlyGeneral(RulesBackend):
        async def decide_task(self, context, facts):
            decision = await super().decide_task(context, facts)
            decision.domain = "general"
            decision.difficulty = 0.5
            decision.domain_confidence = 0.9
            decision.domain_probabilities = {"general": 0.7, "medical": 0.3}
            return decision

    chain = chain_of(config, expensive, MostlyGeneral(config))
    result = await chain.decide_task(EASY, FACTS)

    assert expensive.calls == 1, "a 30% medical share must not be settled by the rules alone"
    assert result.decision.backend == "expensive"
