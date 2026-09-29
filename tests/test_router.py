"""The router: decide, switch, record.

These tests run the real Router over the real rule backend, the real registry
(built from the GGUF files conftest writes) and a real SQLite store. Only
llama-server and the switch executor are fakes, because one is a process and
the other restarts it.

The properties worth pinning are the ones that would be expensive to discover
in production: that a switch happens once even when two requests want it, that
a failed switch still answers, that nothing durable holds a raw prompt, and
that a hard fact beats a model's opinion about a step.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from switchboard.config import Config
from switchboard.decisions.base import StepContext, TaskContext
from switchboard.decisions.schema import StepDecision
from switchboard.router import recommend_next
from switchboard.registry import build_registry
from switchboard.store import Store
from switchboard.system import HostFacts, MemoryStatus

from conftest import DEEP, FAST, FakeExecutor, FakeLlama, build_router

SECRET_PROMPT = "Deploy with OPENAI_API_KEY=sk-abcdefghijklmnopqrstuvwxyz012345 please"


def context(message: str, **kwargs) -> TaskContext:
    kwargs.setdefault("estimated_prompt_tokens", max(1, len(message) // 4))
    return TaskContext(message=message, **kwargs)


async def close(router) -> None:
    await router.chain.aclose()
    await router.upstream.aclose()
    if router.store is not None:
        router.store.close()


# --- registry and resolution ------------------------------------------------


async def test_the_registry_is_built_from_the_stack_on_disk(config: Config) -> None:
    llama = FakeLlama()
    router = build_router(config, llama)
    try:
        snapshot = router.registry(refresh=True)
        assert {model.id for model in snapshot.routable()} == {"fast-9b", "deep-27b", "med-27b"}

        fast = snapshot.by_id("fast-9b")
        assert fast is not None
        assert fast.context_limit == 114688  # from models.json, not the GGUF's 262144
        assert fast.supports_tools is True

        medical = snapshot.by_id("med-27b")
        assert medical is not None
        # Read from the GGUF chat template, not from the overlay's tool_calling.
        assert medical.supports_tools is False
        assert medical.block_count == 62
        assert medical.host_ram_estimate_gb() > 0
    finally:
        await close(router)


def test_unavailable_network_model_does_not_hide_local_models(config: Config, monkeypatch) -> None:
    """A network link can be listed by glob while stat fails after an outage."""
    unavailable = config.stack.models_dir / FAST
    original_stat = Path.stat

    def stat_with_outage(path: Path, *args, **kwargs):
        if path == unavailable:
            raise FileNotFoundError("network share offline")
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", stat_with_outage)
    snapshot = build_registry(config)
    assert snapshot.by_id("fast-9b").exists is False
    assert {model.id for model in snapshot.routable()} == {"deep-27b", "med-27b"}
    assert any("model file unavailable" in problem for problem in snapshot.problems)


async def test_resolve_requested_model(config: Config) -> None:
    llama = FakeLlama()
    router = build_router(config, llama)
    try:
        policy, explicit, recognised = router.resolve_requested_model("switchboard-deep")
        assert (policy, explicit, recognised) == ("deep", None, True)

        policy, explicit, recognised = router.resolve_requested_model("fast9")
        assert explicit is not None and explicit.id == "fast-9b" and recognised

        policy, explicit, recognised = router.resolve_requested_model(FAST)
        assert explicit is not None and explicit.id == "fast-9b"

        policy, explicit, recognised = router.resolve_requested_model("gpt-4o")
        assert explicit is None and recognised is False
    finally:
        await close(router)


async def test_the_resident_model_comes_from_llama_server(config: Config) -> None:
    llama = FakeLlama(resident=DEEP)
    router = build_router(config, llama)
    try:
        status, resident = await router.facts(force=True)
        assert status.healthy
        assert resident is not None and resident.id == "deep-27b"
    finally:
        await close(router)


# --- recording --------------------------------------------------------------


async def test_a_route_is_recorded_without_the_prompt(config: Config) -> None:
    # Previews are off by default; turn them on to prove that even when a
    # preview is kept, the secret inside it is not.
    config.privacy.store_previews = True
    llama = FakeLlama()
    router = build_router(config, llama)
    try:
        plan = await router.route(context(SECRET_PROMPT), source="api")
        assert plan.route_id is not None

        rows = router.store.recent_routes(limit=5)
        assert len(rows) == 1
        row = rows[0]
        assert row["prompt_hash"] and len(row["prompt_hash"]) <= 64
        assert row["selected_model"] == plan.model_id
        assert row["decision_backend"] == "rules"

        # The secret never survives, anywhere in the row.
        blob = json.dumps(row, default=str)
        assert "sk-abcdefghijklmnopqrstuvwxyz012345" not in blob
        assert "REDACTED" in row["preview"]

        # What is kept is a short preview, not the prompt.
        assert len(row["preview"]) <= config.privacy.preview_chars + 1
    finally:
        await close(router)


async def test_previews_can_be_turned_off_entirely(config: Config) -> None:
    config.privacy.store_previews = False
    llama = FakeLlama()
    router = build_router(config, llama)
    try:
        await router.route(context(SECRET_PROMPT), source="api")
        row = router.store.recent_routes(limit=1)[0]
        assert row["preview"] is None
        assert row["prompt_hash"]
    finally:
        await close(router)


async def test_backend_attempts_are_recorded_against_the_route(config: Config) -> None:
    llama = FakeLlama()
    router = build_router(config, llama)
    try:
        await router.route(context("hello there"), source="api")
        stats = router.store.provider_stats(hours=0)
        assert "rules" in stats
        assert stats["rules"]["attempts"] == 1
        assert stats["rules"]["ok"] == 1
    finally:
        await close(router)


async def test_completion_metrics_close_the_loop(config: Config) -> None:
    llama = FakeLlama()
    router = build_router(config, llama)
    try:
        plan = await router.route(context("hello"), source="proxy")
        router.record_completion(
            plan.route_id,
            status_code=200,
            ttfb_ms=120.0,
            total_ms=900.0,
            tokens_in=41,
            tokens_out=64,
            gen_tps=28.5,
            model_id=plan.model_id,
        )
        row = router.store.get_route(plan.route_id)
        assert row["downstream_status"] == 200
        assert row["tokens_out"] == 64

        # A real generation is a real measurement, and the registry picks it up.
        measurements = router.store.measurements_by_model()
        assert measurements[plan.model_id]["gen_tps"].value == pytest.approx(28.5, rel=0.01)
        refreshed = router.registry(refresh=True).by_id(plan.model_id)
        assert refreshed.gen_tps.measured is True
        assert "this machine" in refreshed.gen_tps.source
    finally:
        await close(router)


# --- switching --------------------------------------------------------------


@pytest.mark.parametrize("entrypoint", ["automatic", "named", "direct"])
async def test_every_switch_path_refuses_insufficient_physical_ram(
    config: Config, entrypoint: str
) -> None:
    llama = FakeLlama(resident=FAST)
    executor = FakeExecutor()
    router = build_router(config, llama, executor=executor, free_ram_gb=1.2)
    try:
        deep = router.registry().by_id("deep-27b")
        assert deep is not None
        # The minimal GGUF test file has no real weights. Give it a realistic
        # partial-offload footprint without loading or allocating those weights.
        deep.size_bytes = 13 * 1024**3
        assert deep.host_ram_estimate_gb() == pytest.approx(1.0)

        if entrypoint == "direct":
            result = await router.switch_to(deep)
        else:
            plan = await router.route(
                context("Design a distributed scheduler and justify the trade-offs."),
                policy="deep" if entrypoint == "automatic" else "auto",
                explicit_model=deep if entrypoint == "named" else None,
                requested_model="deep-27b" if entrypoint == "named" else None,
            )
            assert plan.switch is not None
            result = plan.switch

        assert not result.ok
        assert "host RAM" in (result.error or "")
        assert executor.calls == []
    finally:
        await close(router)


async def test_switch_refuses_unknown_ram_but_resident_is_a_noop(config: Config) -> None:
    llama = FakeLlama(resident=FAST)
    executor = FakeExecutor()
    router = build_router(config, llama, executor=executor)
    try:
        router.host._fixed = MemoryStatus(error="GlobalMemoryStatusEx failed")
        deep = router.registry().by_id("deep-27b")
        fast = router.registry().by_id("fast-9b")
        assert deep is not None and fast is not None
        result = await router.switch_to(deep)
        assert not result.ok
        assert "could not be measured" in (result.error or "")
        same = await router.switch_to(fast)
        assert same.ok and same.noop
        assert executor.calls == []
    finally:
        await close(router)


async def test_switch_rechecks_ram_instead_of_reusing_cached_measurement(
    config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    llama = FakeLlama(resident=FAST)
    executor = FakeExecutor()
    router = build_router(config, llama, executor=executor)
    try:
        deep = router.registry().by_id("deep-27b")
        assert deep is not None
        deep.size_bytes = 13 * 1024**3
        readings = iter((MemoryStatus(free_gb=8.0), MemoryStatus(free_gb=1.2)))
        monkeypatch.setattr("switchboard.system.memory_status", lambda: next(readings))
        router._host = HostFacts(ttl_s=3600)
        assert router.host.memory().free_gb == 8.0
        result = await router.switch_to(deep)
        assert not result.ok
        assert executor.calls == []
    finally:
        await close(router)


async def test_fully_gpu_resident_model_can_load_with_headroom(config: Config) -> None:
    llama = FakeLlama(resident=DEEP)
    executor = FakeExecutor()
    router = build_router(config, llama, executor=executor, free_ram_gb=1.1)
    try:
        fast = router.registry().by_id("fast-9b")
        assert fast is not None
        fast.size_bytes = 8 * 1024**3
        assert fast.fully_offloaded and fast.host_ram_estimate_gb() == 0
        result = await router.switch_to(fast)
        assert result.ok
        assert executor.calls == ["fast-9b"]
    finally:
        await close(router)


async def test_an_explicit_model_is_loaded_and_measured(config: Config) -> None:
    llama = FakeLlama(resident=FAST)
    executor = FakeExecutor(load_ms=21_500.0)
    # The fake executor "loads" by changing what llama-server reports.
    executor.on_switch = lambda model_id: setattr(llama, "resident", DEEP)
    router = build_router(config, llama, executor=executor)
    try:
        snapshot = router.registry(refresh=True)
        deep = snapshot.by_id("deep-27b")
        plan = await router.route(
            context("anything"), explicit_model=deep, requested_model="deep-27b", source="api"
        )

        assert executor.calls == ["deep-27b"]
        assert plan.switch is not None and plan.switch.ok
        assert plan.model_id == "deep-27b"

        switches = router.store.recent_switches(limit=5)
        assert switches[0]["to_model"] == "deep-27b"
        assert switches[0]["ok"] is True

        # The measured load replaces the seeded estimate.
        load = router.store.measurements_by_model()["deep-27b"]["load_s"]
        assert load.value == pytest.approx(21.5, rel=0.01)
    finally:
        await close(router)


async def test_a_failed_switch_still_answers_on_the_resident_model(config: Config) -> None:
    llama = FakeLlama(resident=FAST)
    executor = FakeExecutor(ok=False)
    router = build_router(config, llama, executor=executor)
    try:
        deep = router.registry(refresh=True).by_id("deep-27b")
        plan = await router.route(context("anything"), explicit_model=deep, source="api")

        assert executor.calls == ["deep-27b"]
        assert plan.switch is not None and plan.switch.ok is False
        # It answered anyway, on what is actually loaded.
        assert plan.model_id == "fast-9b"
        assert "switch failed" in plan.reason

        failed = router.store.recent_switches(limit=1)[0]
        assert failed["ok"] is False
        assert failed["error"]
    finally:
        await close(router)


async def test_two_concurrent_requests_cause_one_switch(config: Config) -> None:
    """The switch lock. Both want the same model; only one restart happens."""
    llama = FakeLlama(resident=FAST)
    executor = FakeExecutor(load_ms=1000.0)

    def loaded(model_id: str) -> None:
        llama.resident = DEEP

    executor.on_switch = loaded
    router = build_router(config, llama, executor=executor)
    try:
        deep = router.registry(refresh=True).by_id("deep-27b")
        plans = await asyncio.gather(
            router.route(context("first"), explicit_model=deep, source="api"),
            router.route(context("second"), explicit_model=deep, source="api"),
        )

        assert executor.calls == ["deep-27b"], "a second restart would kill the first request"
        assert all(plan.model_id == "deep-27b" for plan in plans)
    finally:
        await close(router)


async def test_a_switch_is_refused_while_a_request_is_in_flight(config: Config) -> None:
    llama = FakeLlama(resident=FAST)
    executor = FakeExecutor()
    router = build_router(config, llama, executor=executor)
    try:
        deep = router.registry(refresh=True).by_id("deep-27b")
        async with router.inflight():
            plan = await router.route(context("anything"), explicit_model=deep, source="api")

        assert executor.calls == []
        assert plan.model_id == "fast-9b"
        assert "in flight" in (plan.switch.error or "")
    finally:
        await close(router)


async def test_shadow_mode_never_switches(config: Config) -> None:
    config.routing.mode = "shadow"
    llama = FakeLlama(resident=FAST)
    executor = FakeExecutor()
    router = build_router(config, llama, executor=executor)
    try:
        plan = await router.route(
            context("Work through the proof of this theorem step by step and justify each step."),
            source="api",
        )
        assert executor.calls == []
        assert plan.model_id == "fast-9b"
    finally:
        await close(router)


# --- sessions ---------------------------------------------------------------


async def test_a_conversation_accumulates_state(config: Config) -> None:
    llama = FakeLlama()
    router = build_router(config, llama)
    try:
        for message in ("hello", "and another thing", "one more"):
            await router.route(context(message, session_id="conv-1"), source="api")

        session = router.store.get_session("conv-1")
        assert session["turns"] == 3
        assert session["current_model"] == "fast-9b"
        assert session["difficulty_ewma"] is not None

        state = router.session_state("conv-1")
        assert state.turns == 3
        assert state.current_model == "fast-9b"
    finally:
        await close(router)


# --- step supervision -------------------------------------------------------


async def test_a_non_zero_exit_code_overrides_a_model_that_says_success() -> None:
    step = StepContext(goal="build", action="cmake --build .", result="Build finished.", exit_code=1)
    decision = StepDecision(outcome="success", progress=0.9, complete=0.9, backend="fixture")

    recommendation, _rationale, overrode = recommend_next(step, decision)

    assert overrode is True
    assert recommendation in {"retry", "retry_different"}


async def test_a_third_failure_escalates_and_names_a_stronger_model(config: Config) -> None:
    llama = FakeLlama(resident=FAST)
    router = build_router(config, llama)
    try:
        evaluation = await router.evaluate_step(
            StepContext(
                goal="fix the failing test",
                action="patch the fixture",
                result="AssertionError: still failing",
                exit_code=1,
                attempt=3,
                previous_attempts=["edit test", "edit fixture"],
            )
        )
        assert evaluation.recommendation == "escalate"
        assert evaluation.escalate_to is not None
        assert evaluation.escalate_to.id == "deep-27b"
        assert "higher capability ceiling" in evaluation.rationale
    finally:
        await close(router)


async def test_escalation_degrades_to_stop_when_nothing_is_stronger(config: Config) -> None:
    llama = FakeLlama(resident=DEEP)
    router = build_router(config, llama)
    try:
        evaluation = await router.evaluate_step(
            StepContext(
                goal="fix it",
                action="try again",
                result="still failing",
                exit_code=1,
                attempt=4,
            )
        )
        assert evaluation.recommendation == "stop_failed"
        assert evaluation.escalate_to is None
        assert "nothing to escalate to" in evaluation.rationale
    finally:
        await close(router)


async def test_a_repeated_action_asks_for_a_different_approach(config: Config) -> None:
    llama = FakeLlama()
    router = build_router(config, llama)
    try:
        evaluation = await router.evaluate_step(
            StepContext(
                goal="install it",
                action="pip install switchboard",
                result="ERROR: Could not find a version that satisfies the requirement",
                exit_code=1,
                attempt=2,
                previous_attempts=["pip install switchboard"],
            )
        )
        assert evaluation.recommendation == "retry_different"
    finally:
        await close(router)


async def test_step_evaluations_are_recorded_by_hash(config: Config) -> None:
    llama = FakeLlama()
    router = build_router(config, llama)
    try:
        await router.evaluate_step(
            StepContext(
                goal="a very secret goal",
                action="run the secret command",
                result="done",
                exit_code=0,
                session_id="conv-9",
            )
        )
        rows = router.store.recent_step_evals(limit=5)
        assert len(rows) == 1
        blob = json.dumps(rows[0], default=str)
        assert "secret" not in blob
        assert rows[0]["recommendation"]
    finally:
        await close(router)


# --- resilience -------------------------------------------------------------


async def test_routing_survives_a_dead_llama_server(config: Config) -> None:
    llama = FakeLlama(healthy=False)
    router = build_router(config, llama)
    try:
        plan = await router.route(context("hello"), source="api")
        # No resident model, nothing in flight: it still produces a plan.
        assert plan.decision.backend == "rules"
        assert plan.route.reason
    finally:
        await close(router)


async def test_routing_survives_an_unwritable_store(config: Config) -> None:
    llama = FakeLlama()
    store = Store(config.db_file)
    store.close()  # every write from here on raises
    router = build_router(config, llama, store=store)
    try:
        plan = await router.route(context("hello"), source="api")
        assert plan.model_id is not None
        assert plan.route_id is None
    finally:
        await router.chain.aclose()
        await router.upstream.aclose()


async def test_status_report_is_complete(config: Config) -> None:
    llama = FakeLlama()
    router = build_router(config, llama)
    try:
        report = await router.status_report()
        assert report["mode"] == "active"
        assert report["upstream"]["healthy"] is True
        assert report["resident_model"] == "fast-9b"
        assert {"jev", "local", "rules"} & set(report["providers"]) or report["providers"]
        assert len(report["models"]) == 3
        assert "ram_free_gb" in report["host"]
    finally:
        await close(router)


async def test_stop_complete_needs_a_high_bar() -> None:
    """Stopping ends the agent loop; a lukewarm "complete" must not do it."""
    step = StepContext(goal="ship it", action="run checks", result="checks passed", exit_code=0)

    lukewarm = StepDecision(outcome="success", progress=0.9, complete=0.65, backend="fixture")
    assert recommend_next(step, lukewarm)[0] == "continue"

    split = StepDecision(
        outcome="success",
        outcome_probabilities={"success": 0.45, "failure": 0.40, "partial": 0.15},
        progress=0.9,
        complete=0.95,
        backend="fixture",
    )
    assert recommend_next(step, split)[0] == "continue"

    clear = StepDecision(
        outcome="success",
        outcome_probabilities={"success": 0.9, "partial": 0.1},
        progress=0.9,
        complete=0.9,
        backend="fixture",
    )
    assert recommend_next(step, clear)[0] == "stop_complete"
