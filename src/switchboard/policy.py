"""The routing policy.

This is the only place where a semantic judgment and a hard fact meet. The
judgment says what kind of work a request is and how hard; the facts say what
is loaded, what fits, how fast each model runs and what a switch costs. The
policy turns both into one choice and one sentence explaining it.

The model is a small expected-utility comparison, deliberately legible:

    utility(m) = value_of_quality_s * P(good answer | m, task) - seconds_of_waiting(m)

`P(good answer)` comes from the model's capability ceiling for the task's
domain against the task's difficulty - a prior from ``config/models.toml``, not
a benchmark. `seconds_of_waiting` is real: measured load time for a switch,
measured tokens/sec for generation. Expressing quality in seconds is what lets
the two be compared at all, and `value_of_quality_s` is the single knob that
says how much a better answer is worth waiting for.

When the judgment arrives as calibrated distributions (Jev), `P(good answer)`
is an expectation: every (domain, difficulty level) cell is scored against the
model's ceiling and weighted by its probability. Collapsing to the top domain
and the mean difficulty first would throw away exactly what the distributions
say - a request that is 45% medical should lean toward the medical model by
that much, and two difficulty distributions with the same mean can deserve
different models because quality is a curve, not a line. Numbers that are not
calibrated probabilities (rule-engine scores, a next-token distribution) are
read as point values only; weighting by them would pass a heuristic off as a
probability.

Switching then has to clear three separate bars, because the expensive mistake
on this machine is a needless 20-second restart:

1. **worth it** - utility must beat staying put by `hysteresis_seconds`;
2. **materially better** - quality must improve by `min_quality_gain`;
3. **allowed right now** - the server is idle, the conversation has held its
   model for `min_dwell_seconds`, the request is not a small utility call, and
   the executor works.

A request that names a model skips all of it: an explicit choice is an
instruction, not a suggestion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from switchboard.config import Config
from switchboard.decisions.schema import DOMAINS, Difficulty, Domain, TaskDecision
from switchboard.registry import ModelEntry, RegistrySnapshot
from switchboard.system import MemoryStatus

Policy = Literal["auto", "fast", "deep", "medical"]

POLICIES: tuple[Policy, ...] = ("auto", "fast", "deep", "medical")

# Virtual model names clients can send, mapped to the policy they select.
VIRTUAL_MODELS: dict[str, Policy] = {
    "switchboard-auto": "auto",
    "switchboard-fast": "fast",
    "switchboard-deep": "deep",
    "switchboard-medical": "medical",
}


@dataclass(slots=True)
class SessionState:
    """What we remember about one conversation. Identified by hash only."""

    session_id: str | None = None
    current_model: str | None = None
    last_switch_ts: float | None = None
    turns: int = 0
    difficulty_ewma: float | None = None

    def updated_difficulty(self, difficulty: float, alpha: float) -> float:
        if self.difficulty_ewma is None:
            return difficulty
        return alpha * difficulty + (1 - alpha) * self.difficulty_ewma


@dataclass(slots=True)
class RouteRequest:
    """Everything about the request that is a fact rather than a judgment."""

    policy: Policy = "auto"
    requested_model: str | None = None
    explicit_model: ModelEntry | None = None
    estimated_prompt_tokens: int = 0
    max_tokens: int | None = None
    streaming: bool = False
    has_tools: bool = False
    session: SessionState = field(default_factory=SessionState)
    # Wall-clock seconds since this process started, for the dwell check.
    now: float = 0.0

    def is_utility_call(self, threshold: int) -> bool:
        """A short non-streaming call: a title, a summary, a classification.

        cptr generates chat titles this way with a 15-second client timeout, so
        making one of these wait for a model load would simply fail.
        """
        return (
            not self.streaming
            and self.max_tokens is not None
            and 0 < self.max_tokens <= threshold
        )


@dataclass(frozen=True, slots=True)
class TaskEstimate:
    """The task as the policy scores it, after the preset has been applied.

    `domain` and `difficulty` are point values: they size the expected answer,
    appear in every reason sentence and are what `RouteDecision` reports.
    Expected quality is summed over `domain_weights` x `difficulty_weights`
    instead. On the point path each holds a single entry of weight 1.0, and the
    sum reduces exactly to the point-value quality, so there is one scoring
    formula rather than two that could drift apart.
    """

    domain: Domain
    difficulty: float
    domain_weights: tuple[tuple[Domain, float], ...]
    # Difficulty levels with the floors already applied, so a level may repeat
    # or be fractional (a session EWMA of 2.8 lifts level 0 to 2.8).
    difficulty_weights: tuple[tuple[float, float], ...]
    # Whether each weight set came from a calibrated distribution; the reason
    # sentence says so when the domain split was used.
    weighted_domain: bool = False
    weighted_difficulty: bool = False

    @classmethod
    def point(cls, domain: Domain, difficulty: float) -> TaskEstimate:
        return cls(
            domain=domain,
            difficulty=difficulty,
            domain_weights=((domain, 1.0),),
            difficulty_weights=((difficulty, 1.0),),
        )


@dataclass(slots=True)
class Candidate:
    """One model scored for one request."""

    model: ModelEntry
    feasible: bool = True
    infeasible_reason: str | None = None
    resident: bool = False
    quality: float = 0.0
    switch_s: float = 0.0
    generate_s: float = 0.0
    prompt_s: float = 0.0
    utility: float = 0.0
    warnings: list[str] = field(default_factory=list)

    @property
    def wait_s(self) -> float:
        return self.switch_s + self.generate_s + self.prompt_s

    def describe(self) -> dict[str, Any]:
        return {
            "model": self.model.id,
            "feasible": self.feasible,
            "infeasible_reason": self.infeasible_reason,
            "resident": self.resident,
            "quality": round(self.quality, 3),
            "switch_s": round(self.switch_s, 1),
            "generate_s": round(self.generate_s, 1),
            "prompt_s": round(self.prompt_s, 1),
            "utility": round(self.utility, 2),
            "warnings": self.warnings,
        }


@dataclass(slots=True)
class RouteDecision:
    """The routing outcome, with enough detail to audit it later."""

    selected: ModelEntry | None
    resident: ModelEntry | None
    switch_required: bool
    switch_allowed: bool
    reason: str
    policy: Policy = "auto"
    explicit: bool = False
    # Set when the policy preferred a model it was not allowed to switch to.
    would_switch_to: ModelEntry | None = None
    blocked_by: str | None = None
    candidates: list[Candidate] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    difficulty_used: float = 0.0
    domain_used: Domain = "general"

    def describe(self) -> dict[str, Any]:
        return {
            "selected": self.selected.id if self.selected else None,
            "resident": self.resident.id if self.resident else None,
            "switch_required": self.switch_required,
            "switch_allowed": self.switch_allowed,
            "would_switch_to": self.would_switch_to.id if self.would_switch_to else None,
            "blocked_by": self.blocked_by,
            "policy": self.policy,
            "explicit": self.explicit,
            "reason": self.reason,
            "difficulty_used": round(self.difficulty_used, 2),
            "domain_used": self.domain_used,
            "warnings": self.warnings,
            "candidates": [candidate.describe() for candidate in self.candidates],
        }


class RoutingPolicy:
    """Scores models for a request and decides whether to move."""

    def __init__(self, config: Config) -> None:
        self._config = config

    # --- component estimates -------------------------------------------

    def quality(self, model: ModelEntry, domain: Domain, difficulty: float) -> float:
        """P(good answer): how far the task sits under the model's ceiling."""
        ceiling = model.capability_for(domain)
        slope = self._config.policy.capability_slope
        return 1.0 / (1.0 + math.exp(-slope * (ceiling - difficulty)))

    def expected_quality(self, model: ModelEntry, estimate: TaskEstimate) -> float:
        """P(good answer) averaged over the task's domain and difficulty weights.

        Each cell is scored at its own level and only then weighted. Scoring the
        mean difficulty instead would interpolate between levels, which a Score
        answer does not license: the quality curve is steep near a ceiling, so
        "half level 2, half level 3" is not the same task as "level 2.5".
        """
        return sum(
            domain_weight * level_weight * self.quality(model, domain, level)
            for domain, domain_weight in estimate.domain_weights
            for level, level_weight in estimate.difficulty_weights
        )

    def expected_output_tokens(self, difficulty: float, max_tokens: int | None) -> int:
        table = self._config.policy.expected_output_tokens
        level = max(0, min(len(table) - 1, round(difficulty)))
        expected = table[level]
        # A client that caps the answer has told us the real length.
        if max_tokens and max_tokens > 0:
            return min(expected, max_tokens)
        return expected

    def switch_seconds(self, model: ModelEntry) -> float:
        """Measured load time, or an estimate derived from file size."""
        if model.load_s is not None:
            return model.load_s.value
        return model.size_gb * self._config.policy.default_load_s_per_gb

    def generate_seconds(self, model: ModelEntry, output_tokens: int) -> float:
        tps = model.gen_tps.value if model.gen_tps else None
        if not tps or tps <= 0:
            # Unknown speed: assume the slowest thing we have measured, so an
            # unmeasured model is never chosen *because* it looks fast.
            tps = 5.0
        return output_tokens / tps

    def prompt_seconds(self, model: ModelEntry, prompt_tokens: int, resident: bool) -> float:
        tps = model.prompt_tps.value if model.prompt_tps else None
        if not tps or tps <= 0:
            tps = 25.0
        seconds = prompt_tokens / tps
        if resident:
            # llama.cpp keeps the conversation's KV cache on the slot (and in the
            # host prompt cache), so a continuing conversation reprocesses little.
            seconds *= 0.25
        return seconds

    # --- feasibility ----------------------------------------------------

    def _check_feasible(
        self,
        model: ModelEntry,
        request: RouteRequest,
        decision: TaskDecision,
        memory: MemoryStatus,
        resident: ModelEntry | None,
        candidate: Candidate,
    ) -> None:
        required = request.estimated_prompt_tokens + self.expected_output_tokens(
            decision.difficulty, request.max_tokens
        )
        limit = model.context_limit
        if limit is not None and required > limit:
            candidate.feasible = False
            candidate.infeasible_reason = (
                f"needs ~{required} tokens of context, {model.id} is configured for {limit}"
            )
            return

        if request.has_tools and not model.supports_tools:
            candidate.feasible = False
            candidate.infeasible_reason = (
                f"{model.id}'s chat template cannot express tool calls, and the request carries tools"
            )
            return

        # Host-RAM risk: the CPU-resident layers of a partial offload are read on
        # every token, and this machine has 15.4 GB total.
        needed = model.host_ram_estimate_gb()
        if needed > 0.5 and memory.known:
            available = (memory.free_gb or 0.0) + (
                resident.host_ram_estimate_gb() if resident and resident.id != model.id else 0.0
            )
            headroom = self._config.routing.guards.ram_headroom_gb
            if needed > available - headroom:
                message = (
                    f"{model.id} needs ~{needed:.1f} GB of host RAM, ~{available:.1f} GB free"
                )
                if self._config.routing.guards.ram_guard == "block":
                    candidate.feasible = False
                    candidate.infeasible_reason = message
                else:
                    candidate.warnings.append(message + " (may page to disk and run far slower)")

    # --- scoring --------------------------------------------------------

    def score(
        self,
        model: ModelEntry,
        request: RouteRequest,
        decision: TaskDecision,
        *,
        resident: ModelEntry | None,
        memory: MemoryStatus,
        estimate: TaskEstimate,
        value_of_quality_s: float,
    ) -> Candidate:
        candidate = Candidate(model=model, resident=bool(resident and resident.id == model.id))
        self._check_feasible(model, request, decision, memory, resident, candidate)

        # Answer length stays on the point difficulty: the token table is a
        # rough per-level guess, and averaging guesses buys no accuracy.
        output_tokens = self.expected_output_tokens(estimate.difficulty, request.max_tokens)
        candidate.quality = self.expected_quality(model, estimate)
        candidate.switch_s = 0.0 if candidate.resident else self.switch_seconds(model)
        candidate.generate_s = self.generate_seconds(model, output_tokens)
        candidate.prompt_s = self.prompt_seconds(
            model, request.estimated_prompt_tokens, candidate.resident
        )
        # Not all waiting is equal. A model load and prompt processing are dead
        # time: nothing happens on screen. Generation on a streamed request is
        # different - the answer starts arriving at once and the reader is busy -
        # so a slower model costs less than its token count suggests. A
        # non-streaming caller does wait for the whole thing, and pays in full.
        weight = self._config.policy.generation_wait_weight if request.streaming else 1.0
        candidate.utility = (
            value_of_quality_s * candidate.quality
            - candidate.switch_s
            - candidate.prompt_s
            - weight * candidate.generate_s
        )
        return candidate

    # --- the decision ---------------------------------------------------

    def route(
        self,
        request: RouteRequest,
        decision: TaskDecision,
        snapshot: RegistrySnapshot,
        *,
        resident: ModelEntry | None,
        memory: MemoryStatus,
        server_busy: bool | None,
        executor_available: bool,
        executor_reason: str = "",
    ) -> RouteDecision:
        """Pick the model this request should run on."""
        estimate, value = self._policy_inputs(request, decision)
        domain, difficulty = estimate.domain, estimate.difficulty

        # An explicit model is an instruction.
        if request.explicit_model is not None:
            model = request.explicit_model
            switch_required = not (resident and resident.id == model.id)
            allowed = (
                switch_required
                and self._config.routing.allow_explicit_switch
                and self._config.routing.mode == "active"
                and executor_available
            )
            if not switch_required:
                reason = f"{model.id} was named by the request and is already loaded."
            elif allowed:
                reason = (
                    f"{model.id} was named by the request; loading it "
                    f"(~{self.switch_seconds(model):.0f}s) instead of routing automatically."
                )
            else:
                # The proxy refuses such a request rather than let another model
                # answer it, so the sentence must not claim a fallback happened.
                reason = (
                    f"{model.id} was named by the request but cannot be loaded "
                    f"({executor_reason or 'switching unavailable'}); an explicit "
                    "choice is not substituted with another model."
                )
            return RouteDecision(
                selected=model if (not switch_required or allowed) else resident,
                resident=resident,
                switch_required=switch_required and allowed,
                switch_allowed=allowed,
                explicit=True,
                policy=request.policy,
                reason=reason,
                would_switch_to=model if (switch_required and not allowed) else None,
                blocked_by=None if (not switch_required or allowed) else "executor",
                difficulty_used=difficulty,
                domain_used=domain,
            )

        candidates = [
            self.score(
                model,
                request,
                decision,
                resident=resident,
                memory=memory,
                estimate=estimate,
                value_of_quality_s=value,
            )
            for model in snapshot.routable()
        ]
        feasible = [candidate for candidate in candidates if candidate.feasible]
        candidates.sort(key=lambda candidate: candidate.utility, reverse=True)

        resident_candidate = next(
            (candidate for candidate in feasible if candidate.resident), None
        )

        if not feasible:
            # Nothing fits. Stay where we are and say why.
            blocked = "; ".join(
                f"{candidate.model.id}: {candidate.infeasible_reason}" for candidate in candidates
            )
            return RouteDecision(
                selected=resident,
                resident=resident,
                switch_required=False,
                switch_allowed=False,
                policy=request.policy,
                reason=f"No model satisfies this request ({blocked}); answering on the resident model.",
                candidates=candidates,
                difficulty_used=difficulty,
                domain_used=domain,
            )

        best = max(feasible, key=lambda candidate: candidate.utility)

        if resident_candidate is not None and best.model.id == resident_candidate.model.id:
            return RouteDecision(
                selected=best.model,
                resident=resident,
                switch_required=False,
                switch_allowed=True,
                policy=request.policy,
                reason=self._stay_reason(best, feasible, estimate),
                candidates=candidates,
                warnings=best.warnings,
                difficulty_used=difficulty,
                domain_used=domain,
            )

        # A move is wanted. Check the guards in order of how much they cost to hit.
        blocked_by, block_reason = self._blocking_guard(
            request, best, resident_candidate, server_busy, executor_available, executor_reason
        )

        if blocked_by is not None:
            fallback = resident_candidate.model if resident_candidate else (resident or best.model)
            return RouteDecision(
                selected=fallback,
                resident=resident,
                switch_required=False,
                switch_allowed=False,
                policy=request.policy,
                reason=block_reason,
                would_switch_to=best.model,
                blocked_by=blocked_by,
                candidates=candidates,
                warnings=best.warnings,
                difficulty_used=difficulty,
                domain_used=domain,
            )

        return RouteDecision(
            selected=best.model,
            resident=resident,
            switch_required=True,
            switch_allowed=True,
            policy=request.policy,
            reason=self._switch_reason(best, resident_candidate, estimate),
            candidates=candidates,
            warnings=best.warnings,
            difficulty_used=difficulty,
            domain_used=domain,
        )

    # --- helpers --------------------------------------------------------

    def _policy_inputs(
        self, request: RouteRequest, decision: TaskDecision
    ) -> tuple[TaskEstimate, float]:
        """The task estimate and quality valuation after the preset is applied.

        The presets act the same way on both paths: the medical preset replaces
        the domain (or the whole domain distribution), and the deep preset and
        the session's running difficulty are floors, applied to the point value
        and to every level of a distribution alike.
        """
        domain: Domain = decision.domain
        value = self._config.policy.value_of_quality_s
        forced_domain: Domain | None = None
        # A conversation that has been hard stays hard: one short follow-up
        # should not drag it back down to the small model and then up again.
        floor = request.session.difficulty_ewma

        if request.policy == "fast":
            # Speed is the point: a better answer is worth only a few seconds.
            value = min(value, 60.0)
        elif request.policy == "deep":
            value = max(value, 1200.0)
            floor = max(floor if floor is not None else 0.0, float(Difficulty.HARD))
        elif request.policy == "medical":
            forced_domain = "medical"
            value = max(value, 900.0)

        difficulty = decision.difficulty if floor is None else max(decision.difficulty, floor)
        if forced_domain is not None:
            domain = forced_domain

        domain_weights: tuple[tuple[Domain, float], ...] | None = None
        level_weights: tuple[tuple[float, float], ...] | None = None
        if decision.calibrated:
            if forced_domain is None:
                domain_weights = _calibrated_domains(decision.domain_probabilities)
            level_weights = _calibrated_levels(decision.difficulty_probabilities, floor)

        estimate = TaskEstimate(
            domain=domain,
            difficulty=difficulty,
            domain_weights=domain_weights or ((domain, 1.0),),
            difficulty_weights=level_weights or ((difficulty, 1.0),),
            weighted_domain=domain_weights is not None,
            weighted_difficulty=level_weights is not None,
        )
        return estimate, value

    def _blocking_guard(
        self,
        request: RouteRequest,
        best: Candidate,
        resident_candidate: Candidate | None,
        server_busy: bool | None,
        executor_available: bool,
        executor_reason: str,
    ) -> tuple[str | None, str]:
        guards = self._config.routing.guards
        target = best.model.id

        if self._config.routing.mode != "active":
            return (
                "mode",
                f"Would route to {target}, but routing.mode is "
                f"{self._config.routing.mode!r}, so the request stays on the loaded model.",
            )

        if not executor_available:
            return "executor", f"Would route to {target}, but {executor_reason or 'no switch executor is available'}."

        if resident_candidate is None:
            # Nothing usable is loaded: a load is the only way to answer at all,
            # so the margin guards do not apply.
            if guards.never_switch_when_busy and server_busy:
                return "busy", f"Would load {target}, but llama-server is generating for another request."
            return None, ""

        gain = best.utility - resident_candidate.utility
        quality_gain = best.quality - resident_candidate.quality

        if request.policy == "fast":
            return (
                "policy",
                f"switchboard-fast keeps the loaded model; {target} would have been the automatic choice.",
            )

        if request.is_utility_call(guards.utility_max_tokens):
            return (
                "utility_call",
                f"Would route to {target}, but this is a short non-streaming call "
                f"(max_tokens={request.max_tokens}); a model load would outlast the client's timeout.",
            )

        if quality_gain < guards.min_quality_gain:
            return (
                "quality_gain",
                f"Stayed on {resident_candidate.model.id}: {target} improves the expected answer by only "
                f"{quality_gain:+.2f} (needs {guards.min_quality_gain:+.2f}).",
            )

        if gain < guards.hysteresis_seconds:
            return (
                "hysteresis",
                f"Stayed on {resident_candidate.model.id}: switching to {target} costs "
                f"~{best.switch_s:.0f}s to load and wins {gain:+.0f}s of expected utility, "
                f"under the {guards.hysteresis_seconds:.0f}s margin.",
            )

        dwell_remaining = self._dwell_remaining(request)
        if dwell_remaining > 0:
            return (
                "dwell",
                f"Stayed on {resident_candidate.model.id}: it was loaded {dwell_remaining:.0f}s "
                f"ago and the minimum dwell is {guards.min_dwell_seconds:.0f}s.",
            )

        if guards.never_switch_when_busy and server_busy:
            return (
                "busy",
                f"Would switch to {target}, but llama-server is generating for another request; "
                f"answering on {resident_candidate.model.id}.",
            )

        return None, ""

    def _dwell_remaining(self, request: RouteRequest) -> float:
        last = request.session.last_switch_ts
        if last is None:
            return 0.0
        elapsed = request.now - last
        return max(0.0, self._config.routing.guards.min_dwell_seconds - elapsed)

    def _stay_reason(
        self, best: Candidate, feasible: list[Candidate], estimate: TaskEstimate
    ) -> str:
        ceiling = best.model.capability_for(estimate.domain)
        others = [candidate for candidate in feasible if candidate.model.id != best.model.id]
        detail = (
            f"Stayed on {best.model.id}: already loaded, and a {estimate.domain} task at "
            f"difficulty {estimate.difficulty:.1f}{_split_note(estimate)} sits within its "
            f"{ceiling:.1f} ceiling."
        )
        if others:
            runner_up = max(others, key=lambda candidate: candidate.utility)
            detail += (
                f" {runner_up.model.id} would cost ~{runner_up.switch_s:.0f}s to load for "
                f"{runner_up.quality - best.quality:+.2f} expected quality."
            )
        return detail

    def _switch_reason(
        self,
        best: Candidate,
        resident_candidate: Candidate | None,
        estimate: TaskEstimate,
    ) -> str:
        domain, difficulty = estimate.domain, estimate.difficulty
        split = _split_note(estimate)
        ceiling = best.model.capability_for(domain)
        if resident_candidate is None:
            return (
                f"Loading {best.model.id}: nothing usable is resident, and its {ceiling:.1f} "
                f"ceiling covers this {domain} task at difficulty {difficulty:.1f}{split}."
            )
        return (
            f"Switching to {best.model.id}: this {domain} task at difficulty {difficulty:.1f}"
            f"{split} is above {resident_candidate.model.id}'s "
            f"{resident_candidate.model.capability_for(domain):.1f} ceiling. Expected quality "
            f"{resident_candidate.quality:.2f} -> {best.quality:.2f} is worth the "
            f"~{best.switch_s:.0f}s load."
        )


# --- calibrated distributions ---------------------------------------------------
#
# Only distributions a backend marks `calibrated` reach these helpers (see
# `RoutingPolicy._policy_inputs`). The API promises probabilities that sum to
# approximately one, so both helpers renormalise; a distribution with no usable
# mass returns None and the policy falls back to the point value.

# A second domain below this share is not worth a mention in the reason.
_SPLIT_NOTE_MIN_SHARE = 0.15


def _calibrated_domains(
    probabilities: dict[str, float],
) -> tuple[tuple[Domain, float], ...] | None:
    weights = [
        (name, float(value))
        for name, value in probabilities.items()
        if name in DOMAINS and value > 0
    ]
    total = sum(value for _, value in weights)
    if not weights or total <= 0:
        return None
    ordered = sorted(weights, key=lambda item: item[1], reverse=True)
    return tuple((cast(Domain, name), value / total) for name, value in ordered)


def _calibrated_levels(
    probabilities: dict[str, float], floor: float | None
) -> tuple[tuple[float, float], ...] | None:
    """Difficulty levels as (level, weight), each lifted to the floor if below it."""
    levels: list[tuple[float, float]] = []
    for key, value in probabilities.items():
        try:
            level = int(key)
        except (TypeError, ValueError):
            continue
        if 0 <= level <= int(Difficulty.HARD) and value > 0:
            lifted = float(level) if floor is None else max(float(level), floor)
            levels.append((lifted, float(value)))
    total = sum(value for _, value in levels)
    if not levels or total <= 0:
        return None
    return tuple((level, value / total) for level, value in levels)


def _split_note(estimate: TaskEstimate) -> str:
    """" (medical 45%, general 55%)" when a calibrated domain split mattered."""
    if not estimate.weighted_domain:
        return ""
    shares = [
        f"{domain} {weight:.0%}"
        for domain, weight in estimate.domain_weights
        if weight >= _SPLIT_NOTE_MIN_SHARE
    ]
    return f" ({', '.join(shares)})" if len(shares) > 1 else ""
