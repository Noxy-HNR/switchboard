"""The eval suite: measure routing instead of trusting it.

Three questions, answered from fixtures rather than intuition:

1. **Do the backends agree with engineering expectation?** Every fixture
   carries a label that says what a competent engineer would call it. The
   labels are expectations, not ground truth, so disagreement is a prompt to
   look at the case - sometimes the fixture is wrong.
2. **What does each backend cost?** Latency percentiles, error rate, tokens
   and, for Jev, dollars. A backend that is 2% more accurate and 40x slower is
   a bad trade for a router that runs before every request.
3. **Does the policy actually hold still?** The switching simulation replays
   fixtures as conversations and counts how often the model would change, with
   the guards on and with them off. That difference is the entire argument for
   hysteresis, and it should be measured, not asserted.

Ranges, not point values: difficulty is scored as a band because a Score answer
is a probability-weighted position, and demanding an exact number would measure
the fixture author's precision rather than the backend's judgment.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from switchboard.config import PROJECT_ROOT, Config, load_config
from switchboard.decisions.base import BackendFacts, StepContext, TaskContext
from switchboard.decisions.chain import DecisionChain
from switchboard.decisions.factory import build_backend
from switchboard.decisions.schema import StepDecision, TaskDecision
from switchboard.policy import RouteRequest, RoutingPolicy, SessionState
from switchboard.registry import RegistrySnapshot, build_registry
from switchboard.router import recommend_next
from switchboard.system import HostFacts, MemoryStatus
from switchboard.upstream import LlamaClient

FIXTURE_DIR = PROJECT_ROOT / "evals" / "fixtures"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports"


# --- fixtures ---------------------------------------------------------------


def load_fixtures(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rows.append(json.loads(line))
        if limit and len(rows) >= limit:
            break
    return rows


def _expected_set(expected: str) -> set[str] | None:
    """`"any"` accepts anything; `"any:a,b"` accepts either."""
    if expected == "any":
        return None
    if expected.startswith("any:"):
        return {item.strip() for item in expected[4:].split(",")}
    return {expected}


def _level_matches(expected: str, value: float) -> bool | None:
    if expected == "any":
        return None
    if expected == "high":
        return value >= 0.5
    if expected == "low":
        return value < 0.5
    return None


# --- scoring ----------------------------------------------------------------


@dataclass(slots=True)
class FixtureResult:
    fixture_id: str
    category: str
    ok: bool
    checks: dict[str, bool | None] = field(default_factory=dict)
    observed: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    backend: str = ""
    error: str | None = None
    # The decision itself, for the switching simulation. `observed` is rounded
    # for the report and drops the distributions a calibrated backend returns,
    # so rebuilding from it would simulate a policy production never runs.
    decision: TaskDecision | None = None

    @property
    def scored_checks(self) -> dict[str, bool]:
        return {name: value for name, value in self.checks.items() if value is not None}

    def describe(self) -> dict[str, Any]:
        return {
            "id": self.fixture_id,
            "category": self.category,
            "ok": self.ok,
            "checks": self.checks,
            "observed": self.observed,
            "latency_ms": round(self.latency_ms, 1),
            "backend": self.backend,
            "error": self.error,
        }


def score_task(fixture: dict[str, Any], decision: TaskDecision) -> dict[str, bool | None]:
    expect = fixture.get("expect", {})
    checks: dict[str, bool | None] = {}

    domains = _expected_set(str(expect.get("domain", "any")))
    checks["domain"] = None if domains is None else decision.domain in domains

    band = expect.get("difficulty")
    if isinstance(band, list) and len(band) == 2:
        checks["difficulty"] = band[0] <= decision.difficulty <= band[1]
    else:
        checks["difficulty"] = None

    checks["tool_need"] = _level_matches(str(expect.get("tool_need", "any")), decision.tool_need)
    checks["ambiguity"] = _level_matches(str(expect.get("ambiguity", "any")), decision.ambiguity)
    return checks


def score_step(
    fixture: dict[str, Any], decision: StepDecision, recommendation: str
) -> dict[str, bool | None]:
    expect = fixture.get("expect", {})
    checks: dict[str, bool | None] = {}

    outcomes = _expected_set(str(expect.get("outcome", "any")))
    checks["outcome"] = None if outcomes is None else decision.outcome in outcomes
    checks["progress"] = _level_matches(str(expect.get("progress", "any")), decision.progress)
    checks["complete"] = _level_matches(str(expect.get("complete", "any")), decision.complete)

    allowed = expect.get("recommendation")
    checks["recommendation"] = recommendation in allowed if isinstance(allowed, list) else None
    return checks


# --- running ----------------------------------------------------------------


@dataclass(slots=True)
class BackendReport:
    backend: str
    kind: str
    results: list[FixtureResult] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    est_cost_usd: float = 0.0
    unavailable: str | None = None

    def summary(self) -> dict[str, Any]:
        latencies = [result.latency_ms for result in self.results if result.error is None]
        scored = [result for result in self.results if result.scored_checks]
        per_check: dict[str, dict[str, int]] = {}
        for result in self.results:
            for name, value in result.checks.items():
                if value is None:
                    continue
                bucket = per_check.setdefault(name, {"pass": 0, "total": 0})
                bucket["total"] += 1
                bucket["pass"] += 1 if value else 0
        return {
            "backend": self.backend,
            "kind": self.kind,
            "unavailable": self.unavailable,
            "fixtures": len(self.results),
            "errors": sum(1 for result in self.results if result.error),
            "agreement": round(
                sum(1 for result in scored if result.ok) / len(scored), 3
            )
            if scored
            else None,
            "per_check": {
                name: {
                    **counts,
                    "rate": round(counts["pass"] / counts["total"], 3) if counts["total"] else None,
                }
                for name, counts in per_check.items()
            },
            "p50_latency_ms": round(statistics.median(latencies), 1) if latencies else None,
            "p95_latency_ms": round(_percentile(latencies, 0.95), 1) if latencies else None,
            "usage": self.usage,
            "est_cost_usd": round(self.est_cost_usd, 6),
        }


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


async def evaluate_backend(
    config: Config,
    backend_name: str,
    task_fixtures: list[dict[str, Any]],
    step_fixtures: list[dict[str, Any]],
    facts: BackendFacts,
) -> tuple[BackendReport, BackendReport]:
    """Run every fixture through one backend in isolation."""
    task_report = BackendReport(backend=backend_name, kind="task")
    step_report = BackendReport(backend=backend_name, kind="step")

    backend = build_backend(backend_name, config)
    if backend is None:
        message = f"{backend_name} could not be constructed"
        task_report.unavailable = step_report.unavailable = message
        return task_report, step_report

    availability = await backend.availability(facts)
    if not availability.ready:
        message = f"{availability.state}: {availability.reason}"
        task_report.unavailable = step_report.unavailable = message
        await backend.aclose()
        return task_report, step_report

    # One backend, no fallback: an eval must measure this backend, not the chain.
    chain = DecisionChain(config, [backend])

    try:
        for fixture in task_fixtures:
            context = TaskContext(
                message=fixture["message"],
                prior_message=fixture.get("prior_message"),
                tool_names=list(fixture.get("tools") or []),
                estimated_prompt_tokens=int(
                    len(fixture["message"]) / config.policy.chars_per_token
                ),
            )
            started = time.perf_counter()
            result = await chain.decide_task(context, facts, use_cache=False)
            elapsed = (time.perf_counter() - started) * 1000
            decision: TaskDecision = result.decision
            error = None if decision.backend == backend_name else _first_error(result)
            checks = score_task(fixture, decision) if error is None else {}
            task_report.results.append(
                FixtureResult(
                    fixture_id=fixture["id"],
                    category=fixture.get("category", ""),
                    ok=all(value for value in checks.values() if value is not None),
                    checks=checks,
                    observed={
                        "domain": decision.domain,
                        "difficulty": round(decision.difficulty, 2),
                        "tool_need": round(decision.tool_need, 2),
                        "ambiguity": round(decision.ambiguity, 2),
                        "confidence": decision.domain_confidence,
                    },
                    latency_ms=elapsed,
                    backend=decision.backend,
                    error=error,
                    decision=decision,
                )
            )
            _accumulate(task_report, decision, config)

        for fixture in step_fixtures:
            step_context = StepContext(
                goal=fixture["goal"],
                action=fixture["action"],
                result=fixture["result"],
                exit_code=fixture.get("exit_code"),
                attempt=int(fixture.get("attempt", 1)),
                previous_attempts=list(fixture.get("previous_attempts") or []),
            )
            started = time.perf_counter()
            step_result = await chain.decide_step(step_context, facts, use_cache=False)
            elapsed = (time.perf_counter() - started) * 1000
            step_decision = step_result.decision
            error = None if step_decision.backend == backend_name else _first_error(step_result)
            recommendation, _, overrode = recommend_next(step_context, step_decision)
            checks = score_step(fixture, step_decision, recommendation) if error is None else {}
            step_report.results.append(
                FixtureResult(
                    fixture_id=fixture["id"],
                    category=fixture.get("category", ""),
                    ok=all(value for value in checks.values() if value is not None),
                    checks=checks,
                    observed={
                        "outcome": step_decision.outcome,
                        "progress": round(step_decision.progress, 2),
                        "complete": round(step_decision.complete, 2),
                        "needs_new_approach": round(step_decision.needs_new_approach, 2),
                        "recommendation": recommendation,
                        "facts_overrode_model": overrode,
                    },
                    latency_ms=elapsed,
                    backend=step_decision.backend,
                    error=error,
                )
            )
            _accumulate(step_report, step_decision, config)
    finally:
        await chain.aclose()

    return task_report, step_report


def _first_error(result: Any) -> str:
    for attempt in result.attempts:
        if not attempt.ok:
            return attempt.error or attempt.skipped_reason or attempt.error_kind or "failed"
    return "no decision"


def _accumulate(report: BackendReport, decision: Any, config: Config) -> None:
    for key, value in (decision.usage or {}).items():
        if isinstance(value, int):
            report.usage[key] = report.usage.get(key, 0) + value
    if report.backend == "jev":
        tokens = decision.usage.get("input_tokens", 0) if decision.usage else 0
        report.est_cost_usd += tokens * config.decisions.jev.usd_per_input_mtok / 1_000_000


# --- switching simulation ---------------------------------------------------


@dataclass(slots=True)
class SwitchSimulation:
    """How often the policy would change models over a fixture sequence."""

    with_guards: int = 0
    without_guards: int = 0
    requests: int = 0
    trace: list[dict[str, Any]] = field(default_factory=list)

    def describe(self) -> dict[str, Any]:
        return {
            "requests": self.requests,
            "switches_with_guards": self.with_guards,
            "switches_without_guards": self.without_guards,
            "switches_per_10_with_guards": round(10 * self.with_guards / self.requests, 2)
            if self.requests
            else None,
            "switches_per_10_without_guards": round(10 * self.without_guards / self.requests, 2)
            if self.requests
            else None,
            "trace": self.trace,
        }


def simulate_switching(
    config: Config,
    snapshot: RegistrySnapshot,
    decisions: list[tuple[dict[str, Any], TaskDecision]],
    memory: MemoryStatus,
) -> SwitchSimulation:
    """Replay decisions as one conversation and count model changes.

    Run twice: once with the configured guards, and once with hysteresis, dwell
    and the quality margin removed, to show what the guards are worth.
    """
    simulation = SwitchSimulation()
    routable = snapshot.routable()
    if not routable:
        return simulation

    start = next((model for model in routable if model.role == "general-fast"), routable[0])

    for guarded in (True, False):
        working = config.model_copy(deep=True)
        if not guarded:
            working.routing.guards.hysteresis_seconds = 0.0
            working.routing.guards.min_quality_gain = 0.0
            working.routing.guards.min_dwell_seconds = 0.0
        policy = RoutingPolicy(working)

        resident = start
        session = SessionState(session_id="simulation")
        clock = 0.0
        switches = 0

        for fixture, decision in decisions:
            clock += 30.0
            request = RouteRequest(
                policy="auto",
                estimated_prompt_tokens=int(
                    len(fixture.get("message", "")) / config.policy.chars_per_token
                ),
                streaming=True,
                has_tools=bool(fixture.get("tools")),
                session=session,
                now=clock,
            )
            route = policy.route(
                request,
                decision,
                snapshot,
                resident=resident,
                memory=memory,
                server_busy=False,
                executor_available=True,
            )
            selected = route.selected or resident
            if selected.id != resident.id:
                switches += 1
                session.last_switch_ts = clock
                if guarded:
                    simulation.trace.append(
                        {
                            "fixture": fixture.get("id"),
                            "from": resident.id,
                            "to": selected.id,
                            "reason": route.reason,
                        }
                    )
                resident = selected
            session.current_model = resident.id
            session.turns += 1
            session.difficulty_ewma = session.updated_difficulty(
                decision.difficulty, config.policy.difficulty_ewma_alpha
            )

        if guarded:
            simulation.with_guards = switches
            simulation.requests = len(decisions)
        else:
            simulation.without_guards = switches

    return simulation


# --- top level --------------------------------------------------------------


async def run_eval(
    config: Config | None = None,
    *,
    backends: Sequence[str] | None = None,
    fixtures_path: Path | None = None,
    steps_path: Path | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Run the full suite and return a machine-readable report."""
    config = config or load_config()
    task_fixtures = load_fixtures(fixtures_path or FIXTURE_DIR / "tasks.jsonl", limit)
    step_fixtures = load_fixtures(steps_path or FIXTURE_DIR / "steps.jsonl", limit)
    names = list(backends or config.decisions.backends)

    # Real facts, so a backend that should refuse (local, while llama-server is
    # busy or slow) refuses here for the same reason it would in production.
    upstream = LlamaClient(config)
    try:
        status = await upstream.status(force=True)
    finally:
        await upstream.aclose()
    snapshot = build_registry(config)
    resident = snapshot.resolve(status.model_path or status.model_id)
    facts = BackendFacts(
        resident_model_id=resident.id if resident else status.model_id,
        resident_gen_tps=resident.gen_tps.value if resident and resident.gen_tps else None,
        server_healthy=status.healthy,
        server_busy=bool(status.busy),
    )

    reports: dict[str, Any] = {}
    simulations: dict[str, Any] = {}
    for name in names:
        task_report, step_report = await evaluate_backend(
            config, name, task_fixtures, step_fixtures, facts
        )
        reports[name] = {"task": task_report.summary(), "step": step_report.summary()}
        reports[name]["task_results"] = [result.describe() for result in task_report.results]
        reports[name]["step_results"] = [result.describe() for result in step_report.results]

        if task_report.results and not task_report.unavailable:
            decisions = [
                (fixture, result.decision)
                for fixture, result in zip(task_fixtures, task_report.results, strict=True)
                if result.error is None and result.decision is not None
            ]
            simulations[name] = simulate_switching(
                config, snapshot, decisions, HostFacts().memory()
            ).describe()

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "machine": {
            "upstream": status.describe(),
            "resident_model": resident.id if resident else None,
            "models": [model.id for model in snapshot.routable()],
        },
        "config": {
            "mode": config.routing.mode,
            "backends": names,
            "budget_ms": config.decisions.budget_ms,
            "guards": config.routing.guards.model_dump(),
            "policy": config.policy.model_dump(),
        },
        "fixtures": {"tasks": len(task_fixtures), "steps": len(step_fixtures)},
        "backends": reports,
        "switching": simulations,
    }


def render_markdown(report: dict[str, Any]) -> str:
    """A human-readable summary of the same numbers."""
    lines = [
        "# Switchboard eval report",
        "",
        f"Generated {report['generated_at']} on this machine.",
        "",
        f"- resident model: `{report['machine']['resident_model']}`",
        f"- models available: {', '.join(report['machine']['models']) or 'none'}",
        f"- fixtures: {report['fixtures']['tasks']} task, {report['fixtures']['steps']} step",
        "",
        "## Backend comparison",
        "",
        "| backend | kind | fixtures | agreement | p50 ms | p95 ms | errors | cost |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, detail in report["backends"].items():
        for kind in ("task", "step"):
            summary = detail[kind]
            if summary.get("unavailable"):
                lines.append(
                    f"| {name} | {kind} | - | _{summary['unavailable']}_ | - | - | - | - |"
                )
                continue
            lines.append(
                f"| {name} | {kind} | {summary['fixtures']} | "
                f"{summary['agreement'] if summary['agreement'] is not None else '-'} | "
                f"{summary['p50_latency_ms']} | {summary['p95_latency_ms']} | "
                f"{summary['errors']} | ${summary['est_cost_usd']:.6f} |"
            )

    lines += ["", "## Per-check agreement", ""]
    for name, detail in report["backends"].items():
        for kind in ("task", "step"):
            summary = detail[kind]
            if summary.get("unavailable") or not summary.get("per_check"):
                continue
            checks = ", ".join(
                f"{check}={counts['pass']}/{counts['total']}"
                for check, counts in summary["per_check"].items()
            )
            lines.append(f"- **{name} / {kind}**: {checks}")

    if report.get("switching"):
        lines += ["", "## Switching simulation", ""]
        lines.append("How often the model would change over the task fixtures replayed as one conversation.")
        lines.append("")
        lines.append("| backend | requests | switches (guards on) | switches (guards off) |")
        lines.append("| --- | ---: | ---: | ---: |")
        for name, simulation in report["switching"].items():
            lines.append(
                f"| {name} | {simulation['requests']} | {simulation['switches_with_guards']} | "
                f"{simulation['switches_without_guards']} |"
            )

    lines += ["", "## Disagreements", ""]
    for name, detail in report["backends"].items():
        for kind in ("task", "step"):
            failures = [
                result
                for result in detail[f"{kind}_results"]
                if not result["ok"] and result["checks"]
            ]
            if not failures:
                continue
            lines.append(f"### {name} / {kind}")
            lines.append("")
            for result in failures:
                missed = [
                    check for check, value in result["checks"].items() if value is False
                ]
                lines.append(
                    f"- `{result['id']}` ({result['category']}): missed {', '.join(missed)}; "
                    f"observed {result['observed']}"
                )
            lines.append("")

    lines += [
        "",
        "_Fixture labels are engineering expectations, not ground truth. A disagreement is a "
        "case to look at, not automatically a defect._",
        "",
    ]
    return "\n".join(lines)


def run_eval_cli(
    *,
    backends: Sequence[str] | None = None,
    fixtures: str | None = None,
    out: str | None = None,
    limit: int | None = None,
) -> int:
    """Entry point for `switchboard eval`. Returns a process exit code."""
    config = load_config()
    report = asyncio.run(
        run_eval(
            config,
            backends=backends,
            fixtures_path=Path(fixtures) if fixtures else None,
            limit=limit,
        )
    )
    directory = Path(out) if out else DEFAULT_REPORT_DIR
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    json_path = directory / f"eval-{stamp}.json"
    md_path = directory / f"eval-{stamp}.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")

    print(render_markdown(report))
    print(f"\nWrote {json_path}\n      {md_path}")

    # Only a hard error fails the command: disagreement is information.
    for detail in report["backends"].values():
        for kind in ("task", "step"):
            if detail[kind]["fixtures"] and detail[kind]["errors"] == detail[kind]["fixtures"]:
                return 1
    return 0
