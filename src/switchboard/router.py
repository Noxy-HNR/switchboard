"""The service layer: decide, switch, record.

Everything with a moving part lives here, so the HTTP layers above stay thin
and the pieces below stay pure. The router owns:

* the registry snapshot, refreshed when the stack's model config changes;
* the decision chain and the routing policy;
* the switch executor, behind a single lock - two requests must never race the
  same llama-server restart;
* conversation state and every measurement the policy later consumes.

The one invariant worth stating: a request is never blocked by Switchboard's
own machinery. Decision providers can be down, the dashboard can be missing,
the database can be unwritable; the request still reaches a model.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from switchboard.config import Config
from switchboard.decisions.base import BackendFacts, StepContext, TaskContext
from switchboard.decisions.chain import DecisionChain, DecisionResult
from switchboard.decisions.schema import Recommendation, StepDecision, TaskDecision
from switchboard.executor import SwitchExecutor, SwitchResult
from switchboard.policy import (
    Policy,
    RouteDecision,
    RouteRequest,
    RoutingPolicy,
    SessionState,
    VIRTUAL_MODELS,
)
from switchboard.registry import Measurement, ModelEntry, RegistrySnapshot, build_registry
from switchboard.store import Store
from switchboard.system import HostFacts
from switchboard.upstream import LlamaClient, UpstreamStatus

logger = logging.getLogger("switchboard.router")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


Source = Literal["proxy", "api", "mcp", "cli", "shadow"]


@dataclass(slots=True)
class RoutedPlan:
    """The outcome of routing one request."""

    model: ModelEntry | None
    route_id: int | None
    decision: TaskDecision
    route: RouteDecision
    switch: SwitchResult | None = None
    chain: DecisionResult | None = None
    session_id: str | None = None

    @property
    def model_id(self) -> str | None:
        return self.model.id if self.model else None

    @property
    def reason(self) -> str:
        if self.switch and not self.switch.ok and self.switch.error:
            return f"{self.route.reason} (switch failed: {self.switch.error})"
        return self.route.reason

    def describe(self) -> dict[str, Any]:
        return {
            "route_id": self.route_id,
            "selected_model": self.model_id,
            "reason": self.reason,
            "decision": self.decision.summary(),
            "routing": self.route.describe(),
            "switch": self.switch.describe() if self.switch else None,
            "providers": self.chain.describe() if self.chain else None,
            "session": self.session_id,
        }


@dataclass(slots=True)
class StepEvaluation:
    """The outcome of evaluating one agent step."""

    recommendation: Recommendation
    rationale: str
    decision: StepDecision
    escalate_to: ModelEntry | None = None
    chain: DecisionResult | None = None
    facts_overrode_model: bool = False

    def describe(self) -> dict[str, Any]:
        return {
            "recommendation": self.recommendation,
            "rationale": self.rationale,
            "escalate_to": self.escalate_to.id if self.escalate_to else None,
            "decision": self.decision.summary(),
            "facts_overrode_model": self.facts_overrode_model,
            "providers": self.chain.describe() if self.chain else None,
        }


def recommend_next(
    context: StepContext, decision: StepDecision
) -> tuple[Recommendation, str, bool]:
    """Turn judgments plus hard facts into one recommendation.

    Facts win over judgments: a non-zero exit code is not an opinion. The
    model's job is the part code cannot see - whether the result actually
    advances the goal.
    """
    overrode = False
    outcome = decision.outcome
    if context.exit_code is not None and context.exit_code != 0 and outcome == "success":
        outcome = "failure"
        overrode = True

    repeated = (
        context.action.strip()
        and context.previous_attempts
        and context.action.strip() in {item.strip() for item in context.previous_attempts}
    )

    # Stopping ends the agent's loop, so a false "complete" is the expensive
    # mistake: the bar is high, and where the backend gave an outcome
    # distribution, success must also hold a clear majority of it.
    success_share = decision.outcome_probabilities.get("success", 1.0)
    if outcome == "success" and decision.complete >= 0.8 and success_share >= 0.6:
        return "stop_complete", (
            f"The step succeeded and the goal looks satisfied (complete={decision.complete:.2f})."
        ), overrode

    if outcome == "success":
        return "continue", (
            f"The step succeeded and made progress (progress={decision.progress:.2f}); "
            "carry on with the next step."
        ), overrode

    if outcome == "partial":
        if decision.progress >= 0.5:
            return "continue", (
                f"Partial success with real progress (progress={decision.progress:.2f}); "
                "continue from where it got to."
            ), overrode
        return "retry_different", (
            "Partial result without meaningful progress; the current approach is not getting there."
        ), overrode

    if outcome == "failure":
        attempts = max(1, context.attempt)
        if attempts >= 3:
            return "escalate", (
                f"Attempt {attempts} failed. Repeating this is unlikely to help."
            ), overrode
        if decision.needs_new_approach >= 0.6 or repeated:
            reason = (
                "the same action was already tried"
                if repeated
                else "running it again unchanged would fail the same way "
                f"(needs_new_approach={decision.needs_new_approach:.2f})"
            )
            return "retry_different", f"The step failed and {reason}.", overrode
        exit_note = (
            f" (exit code {context.exit_code})" if context.exit_code not in (None, 0) else ""
        )
        return "retry", (
            f"The step failed{exit_note} but the cause looks transient, so a plain "
            f"retry may clear it; attempt {attempts} of 3."
        ), overrode

    # unclear
    if context.attempt >= 2:
        return "ask_user", (
            "The result does not say whether the step worked, and this is not the first attempt; "
            "ask the user rather than guessing."
        ), overrode
    return "continue", (
        "The result is inconclusive; continue and let the next step disambiguate."
    ), overrode


class Router:
    """Ties the decision layer, the policy and the executor together."""

    def __init__(
        self,
        config: Config,
        *,
        upstream: LlamaClient,
        chain: DecisionChain,
        executor: SwitchExecutor,
        store: Store | None = None,
        host: HostFacts | None = None,
    ) -> None:
        self._config = config
        self._upstream = upstream
        self._chain = chain
        self._executor = executor
        self._store = store
        self._host = host or HostFacts()
        self._policy = RoutingPolicy(config)
        self._switch_lock = asyncio.Lock()
        self._inflight = 0
        self._snapshot: RegistrySnapshot | None = None
        self._snapshot_signature: tuple[float, int] | None = None
        self._started = time.monotonic()

    # --- registry -------------------------------------------------------

    @property
    def config(self) -> Config:
        return self._config

    @property
    def chain(self) -> DecisionChain:
        return self._chain

    @property
    def executor(self) -> SwitchExecutor:
        return self._executor

    @property
    def store(self) -> Store | None:
        return self._store

    @property
    def upstream(self) -> LlamaClient:
        return self._upstream

    @property
    def host(self) -> HostFacts:
        return self._host

    def registry(self, *, refresh: bool = False) -> RegistrySnapshot:
        """The current registry, rebuilt when the dashboard's config changes."""
        signature = self._config_signature()
        if refresh or self._snapshot is None or signature != self._snapshot_signature:
            measurements = self._measurements()
            self._snapshot = build_registry(self._config, measurements=measurements)
            self._snapshot_signature = signature
        return self._snapshot

    def _config_signature(self) -> tuple[float, int]:
        """Cheap change detector for C:\\AI\\Config\\models.json."""
        path = self._config.stack.models_config
        try:
            stat = path.stat()
        except OSError:
            return (0.0, 0)
        return (stat.st_mtime, stat.st_size)

    def _measurements(self) -> dict[str, dict[str, Measurement]]:
        if self._store is None:
            return {}
        try:
            return self._store.measurements_by_model()
        except Exception as error:  # noqa: BLE001 - observability must not break routing
            logger.warning("could not read measurements: %s", error)
            return {}

    def invalidate_registry(self) -> None:
        self._snapshot_signature = None

    # --- facts ----------------------------------------------------------

    async def facts(self, *, force: bool = False) -> tuple[UpstreamStatus, ModelEntry | None]:
        """Upstream status plus the registry entry for whatever is resident."""
        status = await self._upstream.status(force=force)
        snapshot = self.registry()
        resident = snapshot.resolve(status.model_path or status.model_id) if status.healthy else None
        return status, resident

    async def resident_now(self) -> tuple[str | None, ModelEntry | None]:
        """The model loaded at this instant, read fresh with a single request.

        Used under the in-flight reservation just before generating, where the
        cached status could be two seconds stale and a full refresh would add
        three calls nobody needs.
        """
        model_id = await self._upstream.resident_model_id()
        return model_id, self.registry().resolve(model_id) if model_id else None

    def backend_facts(self, status: UpstreamStatus, resident: ModelEntry | None) -> BackendFacts:
        return BackendFacts(
            resident_model_id=resident.id if resident else status.model_id,
            resident_gen_tps=(resident.gen_tps.value if resident and resident.gen_tps else None),
            server_healthy=status.healthy,
            server_busy=bool(status.busy) or self._inflight > 0,
        )

    # --- sessions -------------------------------------------------------

    def session_state(self, session_id: str | None) -> SessionState:
        state = SessionState(session_id=session_id)
        if not session_id or self._store is None:
            return state
        try:
            row = self._store.get_session(session_id)
        except Exception as error:  # noqa: BLE001
            logger.warning("could not read session %s: %s", session_id, error)
            return state
        if not row:
            return state
        state.current_model = row.get("current_model")
        state.turns = int(row.get("turns") or 0)
        difficulty = row.get("difficulty_ewma")
        state.difficulty_ewma = float(difficulty) if difficulty is not None else None
        last_switch = row.get("last_switch_ts")
        state.last_switch_ts = self._monotonic_from_stored(last_switch)
        return state

    def _monotonic_from_stored(self, value: Any) -> float | None:
        """Translate a stored wall-clock switch time into this process's clock.

        Dwell only needs elapsed seconds, and mixing clocks across restarts is a
        classic source of nonsense, so a switch recorded before this process
        started is simply treated as long ago.
        """
        if value is None:
            return None
        try:
            stamp = datetime.fromisoformat(str(value))
            if stamp.tzinfo is None:
                stamp = stamp.replace(tzinfo=timezone.utc)
            age = (datetime.now(timezone.utc) - stamp).total_seconds()
        except (TypeError, ValueError):
            return None
        return time.monotonic() - max(0.0, age)

    def _persist_session(
        self, session_id: str | None, plan_model: str | None, decision: TaskDecision, switched: bool
    ) -> None:
        if not session_id or self._store is None:
            return
        state = self.session_state(session_id)
        fields: dict[str, Any] = {
            "current_model": plan_model,
            "difficulty_ewma": state.updated_difficulty(
                decision.difficulty, self._config.policy.difficulty_ewma_alpha
            ),
        }
        if switched:
            fields["last_switch_ts"] = _now_iso()
        with contextlib.suppress(Exception):
            self._store.upsert_session(session_id, bump_turns=True, **fields)

    # --- routing --------------------------------------------------------

    def resolve_requested_model(
        self, requested: str | None
    ) -> tuple[Policy, ModelEntry | None, bool]:
        """Interpret the request's `model` field.

        Returns the policy to apply, an explicitly named model if there was one,
        and whether the name was recognised at all.
        """
        snapshot = self.registry()
        if not requested:
            return self._config.routing.default_policy, None, True  # type: ignore[return-value]
        name = requested.strip()
        lowered = name.lower()
        if lowered in VIRTUAL_MODELS:
            return VIRTUAL_MODELS[lowered], None, True
        entry = snapshot.resolve(name)
        if entry is not None:
            return "auto", entry, True
        return self._config.routing.default_policy, None, False  # type: ignore[return-value]

    async def route(
        self,
        context: TaskContext,
        *,
        policy: Policy = "auto",
        explicit_model: ModelEntry | None = None,
        requested_model: str | None = None,
        max_tokens: int | None = None,
        streaming: bool = False,
        source: Source = "proxy",
        allow_switch: bool = True,
        use_cache: bool = True,
    ) -> RoutedPlan:
        """Decide, switch if warranted, and record the result."""
        status, resident = await self.facts()
        facts = self.backend_facts(status, resident)

        if explicit_model is not None or self._config.routing.mode == "off":
            chain_result = DecisionResult(decision=TaskDecision(backend="bypass"))
        else:
            chain_result = await self._chain.decide_task(context, facts, use_cache=use_cache)
        decision: TaskDecision = chain_result.decision
        resident_before = resident.id if resident else status.model_id

        executor_available, executor_reason = await self._executor.available()
        if not allow_switch:
            executor_available = False
            executor_reason = "this request opted out of switching"

        request = RouteRequest(
            policy=policy,
            requested_model=requested_model,
            explicit_model=explicit_model,
            estimated_prompt_tokens=context.estimated_prompt_tokens,
            max_tokens=max_tokens,
            streaming=streaming,
            has_tools=bool(context.tool_names),
            session=self.session_state(context.session_id),
            now=time.monotonic(),
        )

        route = self._policy.route(
            request,
            decision,
            self.registry(),
            resident=resident,
            memory=self._host.memory(),
            server_busy=facts.server_busy,
            executor_available=executor_available,
            executor_reason=executor_reason,
        )

        switch: SwitchResult | None = None
        if route.switch_required and route.selected is not None:
            switch = await self._perform_switch(route.selected, resident)
            if not switch.ok:
                # The switch failed; answer on whatever is actually loaded.
                status, resident = await self.facts(force=True)
                route.selected = resident or route.selected
                route.switch_required = False

        route_id = self._record_route(
            context=context,
            decision=decision,
            route=route,
            chain=chain_result,
            switch=switch,
            source=source,
            streaming=streaming,
            resident_before=resident_before,
            requested_model=requested_model,
        )

        if allow_switch:
            self._persist_session(
                context.session_id,
                route.selected.id if route.selected else None,
                decision,
                switched=bool(switch and switch.ok),
            )

        return RoutedPlan(
            model=route.selected,
            route_id=route_id,
            decision=decision,
            route=route,
            switch=switch,
            chain=chain_result,
            session_id=context.session_id,
        )

    async def _perform_switch(
        self, target: ModelEntry, resident: ModelEntry | None
    ) -> SwitchResult:
        """An automatic switch, decided by the policy."""
        return await self._guarded_switch(target, trigger="auto")

    async def switch_to(self, target: ModelEntry, *, trigger: str = "manual") -> SwitchResult:
        """An explicit switch, from the CLI or the agent API."""
        return await self._guarded_switch(target, trigger=trigger)

    async def _guarded_switch(self, target: ModelEntry, *, trigger: str) -> SwitchResult:
        """The one path to a restart, with every guard, under the lock.

        Automatic and explicit switches share it deliberately: they used to have
        separate copies of these checks, and the copies had already drifted.
        """
        async with self._switch_lock:
            # Re-check under the lock: another request may have just loaded it.
            status, current = await self.facts(force=True)
            from_model = current.id if current else None

            def refused(reason: str) -> SwitchResult:
                return SwitchResult(
                    ok=False,
                    executor=self._executor.name,
                    from_model=from_model,
                    to_model=target.id,
                    error=reason,
                )

            if current and current.id == target.id:
                return SwitchResult(
                    ok=True,
                    executor=self._executor.name,
                    from_model=current.id,
                    to_model=target.id,
                    noop=True,
                )
            if not target.enabled or not target.exists:
                return refused(f"{target.id} is disabled or its file is missing")
            if self._inflight > 0:
                return refused(f"{self._inflight} request(s) in flight through Switchboard")
            # A healthy server whose slot state is unknown is treated as busy: a
            # restart drops whatever it is doing, and "unknown" is not "idle".
            # A server that is down has nothing to drop, so a load may proceed.
            if status.healthy and status.busy is not False:
                return refused("llama-server is busy or its slot state is unknown")

            result = await self._run_to_completion(target, from_model, trigger=trigger)
            self._record_switch(result, trigger=trigger)
            if result.ok and result.load_ms:
                source = "switch" if trigger == "auto" else f"switch:{trigger}"
                self._record_measurement(
                    target.id, "load_s", result.load_ms / 1000.0, source=source
                )
                self.invalidate_registry()
            return result

    async def _run_to_completion(
        self, target: ModelEntry, from_model: str | None, *, trigger: str
    ) -> SwitchResult:
        """Run the executor, holding the lock until the restart really ends.

        The executor asks the dashboard to restart llama-server, which carries on
        whether or not our caller is still waiting. If the caller is cancelled
        (a client disconnects mid-load) and the lock were released at once, the
        next request would start generating against a server that is still
        coming up with a different model. So the external action is shielded,
        the lock stays held until it finishes, and only then is the cancellation
        passed on.
        """
        task = asyncio.ensure_future(self._executor.switch(target, from_model=from_model))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            with contextlib.suppress(Exception):
                # The switch happened even though nobody is waiting for it.
                self._record_switch(await task, trigger=trigger)
            raise

    # --- step supervision -----------------------------------------------

    async def evaluate_step(self, context: StepContext, *, use_cache: bool = True) -> StepEvaluation:
        """Judge a finished agent step and recommend what to do next."""
        status, resident = await self.facts()
        facts = self.backend_facts(status, resident)
        chain_result = await self._chain.decide_step(context, facts, use_cache=use_cache)
        decision: StepDecision = chain_result.decision

        recommendation, rationale, overrode = recommend_next(context, decision)
        escalate_to = None
        if recommendation == "escalate":
            escalate_to = self._stronger_than(resident)
            if escalate_to is None:
                recommendation = "stop_failed"
                rationale += " No stronger local model is available, so there is nothing to escalate to."
            else:
                rationale += f" {escalate_to.id} has a higher capability ceiling than {resident.id if resident else 'the resident model'}."

        self._record_step(context, decision, recommendation, chain_result)
        return StepEvaluation(
            recommendation=recommendation,
            rationale=rationale,
            decision=decision,
            escalate_to=escalate_to,
            chain=chain_result,
            facts_overrode_model=overrode,
        )

    def _stronger_than(self, resident: ModelEntry | None) -> ModelEntry | None:
        """The best general-capability model other than the one in use."""
        candidates = [
            model
            for model in self.registry().routable()
            if resident is None or model.id != resident.id
        ]
        if not candidates:
            return None
        best = max(candidates, key=lambda model: model.capability_for("general"))
        if resident and best.capability_for("general") <= resident.capability_for("general"):
            return None
        return best

    # --- in-flight accounting -------------------------------------------

    @contextlib.asynccontextmanager
    async def inflight(self):
        """Count a request as in flight so a switch cannot interrupt it."""
        # A generation cannot begin during a restart. Switches inspect the count
        # under this same lock, including the explicit agent switch endpoint.
        async with self._switch_lock:
            self._inflight += 1
        try:
            yield
        finally:
            self._inflight = max(0, self._inflight - 1)

    @property
    def inflight_count(self) -> int:
        return self._inflight

    @property
    def uptime_s(self) -> float:
        return time.monotonic() - self._started

    # --- recording ------------------------------------------------------

    def _record_route(
        self,
        *,
        context: TaskContext,
        decision: TaskDecision,
        route: RouteDecision,
        chain: DecisionResult,
        switch: SwitchResult | None,
        source: Source,
        streaming: bool,
        resident_before: str | None,
        requested_model: str | None,
    ) -> int | None:
        if self._store is None:
            return None
        from switchboard.privacy import hash_text, preview

        privacy = self._config.privacy
        try:
            route_id = self._store.record_route(
                session_id=context.session_id,
                source=source,
                mode=self._config.routing.mode,
                policy=route.policy,
                requested_model=requested_model,
                selected_model=route.selected.id if route.selected else None,
                resident_before=resident_before,
                switched=bool(switch and switch.ok and not switch.noop),
                switch_ms=switch.load_ms if switch else None,
                decision_backend=decision.backend,
                decision_latency_ms=decision.latency_ms,
                decision=decision.summary(),
                reason=route.reason,
                confidence=decision.domain_confidence,
                prompt_hash=hash_text(context.message),
                preview=(
                    preview(context.message, limit=privacy.preview_chars)
                    if privacy.store_previews
                    else None
                ),
                estimated_prompt_tokens=context.estimated_prompt_tokens,
                streaming=streaming,
            )
        except Exception as error:  # noqa: BLE001
            logger.warning("could not record route: %s", error)
            return None
        self._record_attempts(chain, kind="task", route_id=route_id)
        return route_id

    def _record_attempts(self, chain: DecisionResult, *, kind: str, route_id: int | None) -> None:
        """Store one row per backend attempt, with Jev's cost where it applies."""
        if self._store is None:
            return
        for attempt in chain.attempts:
            usage = attempt.usage or {}
            input_tokens = usage.get("input_tokens")
            cost = usage.get("est_cost_usd")
            if cost is None and "est_cost_usd_micro" in usage:
                cost = usage["est_cost_usd_micro"] / 1_000_000
            if cost is None and attempt.backend == "jev" and input_tokens:
                cost = input_tokens * self._config.decisions.jev.usd_per_input_mtok / 1_000_000
            with contextlib.suppress(Exception):
                self._store.record_attempt(
                    backend=attempt.backend,
                    kind=kind,  # type: ignore[arg-type]
                    ok=attempt.ok,
                    route_id=route_id,
                    error_kind=attempt.error_kind,
                    latency_ms=attempt.latency_ms,
                    input_tokens=input_tokens,
                    output_tokens=usage.get("output_tokens"),
                    est_cost_usd=cost,
                )

    def record_completion(
        self,
        route_id: int | None,
        *,
        status_code: int | None = None,
        ttfb_ms: float | None = None,
        total_ms: float | None = None,
        tokens_in: int | None = None,
        tokens_out: int | None = None,
        gen_tps: float | None = None,
        error: str | None = None,
        model_id: str | None = None,
    ) -> None:
        """Close the loop on a routed request once its response is done."""
        if self._store is None:
            return
        with contextlib.suppress(Exception):
            if route_id is not None:
                self._store.update_route(
                    route_id,
                    downstream_status=status_code,
                    ttfb_ms=ttfb_ms,
                    downstream_ms=total_ms,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    gen_tps=gen_tps,
                    error=error,
                )
        # A real generation is the best tokens/sec measurement there is.
        if model_id and gen_tps and gen_tps > 0 and (tokens_out or 0) >= 32:
            self._record_measurement(model_id, "gen_tps", gen_tps, source="traffic")

    def _record_measurement(self, model_id: str, metric: str, value: float, *, source: str) -> None:
        if self._store is None:
            return
        with contextlib.suppress(Exception):
            self._store.record_measurement(
                model_id=model_id, metric=metric, value=value, source=source
            )
        self.invalidate_registry()

    def _record_switch(self, result: SwitchResult, *, trigger: str) -> None:
        if self._store is None or result.noop:
            return
        with contextlib.suppress(Exception):
            self._store.record_switch(
                from_model=result.from_model,
                to_model=result.to_model,
                executor=result.executor,
                trigger=trigger,
                reason=None,
                ok=result.ok,
                load_ms=result.load_ms,
                error=result.error,
            )

    def _record_step(
        self,
        context: StepContext,
        decision: StepDecision,
        recommendation: Recommendation,
        chain: DecisionResult,
    ) -> None:
        if self._store is None:
            return
        from switchboard.privacy import hash_text

        with contextlib.suppress(Exception):
            self._store.record_step_eval(
                session_id=context.session_id,
                goal_hash=hash_text(context.goal),
                action_hash=hash_text(context.action),
                attempt=context.attempt,
                outcome=decision.outcome,
                progress=decision.progress,
                complete=decision.complete,
                needs_new_approach=decision.needs_new_approach,
                recommendation=recommendation,
                backend=decision.backend,
                latency_ms=decision.latency_ms,
            )
        self._record_attempts(chain, kind="step", route_id=None)

    # --- status ---------------------------------------------------------

    async def status_report(self) -> dict[str, Any]:
        """Everything /status and `switchboard status` need, in one place."""
        status, resident = await self.facts()
        snapshot = self.registry()
        executor_ok, executor_reason = await self._executor.available()
        return {
            "mode": self._config.routing.mode,
            "uptime_s": round(self.uptime_s, 1),
            "inflight": self._inflight,
            "upstream": status.describe(),
            "resident_model": resident.id if resident else None,
            "executor": {
                "name": self._executor.name,
                "available": executor_ok,
                "detail": executor_reason,
            },
            "providers": self._chain.health(),
            "models": [model.describe() for model in snapshot.models],
            "registry_problems": snapshot.problems,
            "host": await self._host.describe(),
        }
