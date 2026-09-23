"""Tests for the eval runner: what it measures must be what production runs."""

from __future__ import annotations

from typing import Any

import pytest

from switchboard.config import Config
from switchboard.decisions.base import Availability, BackendFacts, StepContext, TaskContext
from switchboard.decisions.schema import StepDecision, TaskDecision
from switchboard.evals import runner

FIXTURES: list[dict[str, Any]] = [
    {"id": "one", "category": "medical", "message": "Explain my blood test.", "expect": {"domain": "medical"}},
    {"id": "two", "category": "general", "message": "Say hello.", "expect": {"domain": "general"}},
]


class CalibratedBackend:
    """Answers like Jev: calibrated distributions and an unrounded difficulty."""

    name = "calibrated"

    async def availability(self, facts: BackendFacts) -> Availability:
        return Availability.ok()

    async def decide_task(self, context: TaskContext, facts: BackendFacts) -> TaskDecision:
        return TaskDecision(
            domain="general",
            difficulty=1.2345,
            domain_probabilities={"general": 0.55, "medical": 0.45},
            difficulty_probabilities={"0": 0.1, "1": 0.6, "2": 0.3},
            calibrated=True,
            backend=self.name,
        )

    async def decide_step(self, context: StepContext, facts: BackendFacts) -> StepDecision:
        return StepDecision(outcome="success", backend=self.name)

    async def aclose(self) -> None:
        return None


async def test_the_switching_simulation_sees_the_decision_production_would(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The simulation used to rebuild each decision from the rounded report
    fields, which drop the calibrated distributions the policy weighs, so it
    simulated a policy production never runs for a calibrated backend."""
    monkeypatch.setattr(runner, "build_backend", lambda name, config: CalibratedBackend())

    task_report, _ = await runner.evaluate_backend(Config(), "calibrated", FIXTURES, [], BackendFacts())

    decisions = [result.decision for result in task_report.results]
    assert all(decision is not None and decision.calibrated for decision in decisions)
    assert decisions[0].domain_probabilities == {"general": 0.55, "medical": 0.45}
    assert decisions[0].difficulty == 1.2345  # the report rounds it; the simulation must not
    # The report itself stays as it was: the decision object is not serialised.
    assert "decision" not in task_report.results[0].describe()
