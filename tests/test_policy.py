"""The routing policy.

This is where a semantic judgment meets a hard fact, so these tests are the
evidence for the claims the project actually makes: that a stronger model does
not automatically win, that a conversation holds still, and that the three
guards are the difference between routing and thrashing.

Models are built by hand here rather than read from disk. The registry's own
reading is covered elsewhere; what matters here is the arithmetic, and a
hand-built model lets a test say "this one is slow and strong" in one line.
"""

from __future__ import annotations

import pytest

from switchboard.config import Config
from switchboard.decisions.schema import TaskDecision
from switchboard.policy import RouteRequest, RoutingPolicy, SessionState
from switchboard.recording import RecordingState
from switchboard.registry import Measurement, ModelEntry, RegistrySnapshot
from switchboard.system import MemoryStatus

PLENTY_OF_RAM = MemoryStatus(total_gb=15.4, free_gb=9.0)
ALMOST_NO_RAM = MemoryStatus(total_gb=15.4, free_gb=1.2)


def model(
    identifier: str,
    *,
    general: float = 2.0,
    coding: float = 2.0,
    medical: float = 1.0,
    writing: float = 2.0,
    research: float = 2.0,
    system_ops: float = 2.0,
    gen_tps: float | None = 30.0,
    load_s: float | None = 12.0,
    context: int = 114688,
    tools: bool = True,
    size_gb: float = 8.0,
    gpu_layers: int = 32,
    block_count: int = 32,
    role: str = "general-fast",
) -> ModelEntry:
    return ModelEntry(
        id=identifier,
        filename=f"{identifier}.gguf",
        path=__import__("pathlib").Path(f"C:/fake/{identifier}.gguf"),
        exists=True,
        enabled=True,
        role=role,
        configured_context=context,
        gpu_layers=gpu_layers,
        block_count=block_count,
        size_bytes=int(size_gb * 1024**3),
        template_supports_tools=tools,
        capability={
            "general": general,
            "coding": coding,
            "medical": medical,
            "writing": writing,
            "research": research,
            "system_ops": system_ops,
        },
        gen_tps=None if gen_tps is None else Measurement(gen_tps, "fixture"),
        prompt_tps=Measurement(300.0, "fixture"),
        load_s=None if load_s is None else Measurement(load_s, "fixture"),
    )


FAST = model("fast-9b", general=2.1, coding=2.0, medical=1.0, gen_tps=30.0, load_s=12.0)
DEEP = model(
    "deep-27b",
    general=2.6,
    coding=2.5,
    medical=1.6,
    research=2.5,
    system_ops=2.4,
    gen_tps=10.0,
    load_s=22.0,
    context=20480,
    size_gb=11.9,
    gpu_layers=60,
    block_count=65,
    role="deep-reasoning",
)
MED = model(
    "med-27b",
    general=1.9,
    coding=1.2,
    medical=2.8,
    gen_tps=5.0,
    load_s=27.6,
    context=32768,
    tools=False,
    size_gb=15.4,
    gpu_layers=31,
    block_count=62,
    role="medical-specialist",
)


def snapshot(*models: ModelEntry) -> RegistrySnapshot:
    return RegistrySnapshot(models=list(models or (FAST, DEEP, MED)))


def decision(domain: str = "general", difficulty: float = 1.0, **kwargs) -> TaskDecision:
    return TaskDecision(domain=domain, difficulty=difficulty, backend="rules", **kwargs)


def route(
    config: Config,
    task: TaskDecision,
    *,
    resident: ModelEntry | None = FAST,
    models: tuple[ModelEntry, ...] = (),
    memory: MemoryStatus = PLENTY_OF_RAM,
    busy: bool = False,
    executor_available: bool = True,
    recording: RecordingState | None = None,
    **request_kwargs,
):
    policy = RoutingPolicy(config)
    # Chat traffic streams. Non-streaming callers pay the full generation cost,
    # which changes the arithmetic, so tests about that say so explicitly.
    request_kwargs.setdefault("streaming", True)
    request = RouteRequest(**request_kwargs)
    return policy.route(
        request,
        task,
        snapshot(*models),
        resident=resident,
        memory=memory,
        server_busy=busy,
        executor_available=executor_available,
        executor_reason="fake executor",
        recording=recording,
    )


# --- staying put ------------------------------------------------------------


def test_easy_task_stays_on_the_resident_model() -> None:
    config = Config()
    result = route(config, decision("general", 0.5))

    assert result.selected is FAST
    assert result.switch_required is False
    assert "fast-9b" in result.reason
    # The sentence has to say why, in facts: the ceiling and the task's difficulty.
    assert "2.1" in result.reason and "0.5" in result.reason


def test_a_stronger_model_does_not_automatically_win() -> None:
    """The whole point: deep-27b is better at everything and still loses here."""
    config = Config()
    result = route(config, decision("coding", 1.2))

    assert result.selected is FAST
    assert result.would_switch_to is None or result.would_switch_to is DEEP
    assert result.switch_required is False


def test_hard_task_escalates_to_the_deeper_model() -> None:
    config = Config()
    result = route(config, decision("general", 2.9))

    assert result.selected is DEEP
    assert result.switch_required is True
    assert "22" in result.reason  # the load it is paying for


# --- the three guards -------------------------------------------------------


def test_quality_gain_guard_blocks_a_marginal_switch() -> None:
    config = Config()
    # Difficulty just above the fast model's ceiling: a real but small gain.
    config.routing.guards.min_quality_gain = 0.9  # nothing can clear this
    result = route(config, decision("general", 2.6))

    assert result.switch_required is False
    assert result.blocked_by == "quality_gain"
    assert result.would_switch_to is DEEP
    assert "needs" in result.reason


def test_hysteresis_blocks_a_switch_that_does_not_pay_for_itself() -> None:
    config = Config()
    config.routing.guards.min_quality_gain = 0.0
    config.routing.guards.hysteresis_seconds = 10_000.0
    result = route(config, decision("general", 2.9))

    assert result.switch_required is False
    assert result.blocked_by == "hysteresis"
    assert result.would_switch_to is DEEP
    assert "margin" in result.reason


def test_dwell_keeps_a_freshly_loaded_model_for_a_while() -> None:
    config = Config()
    session = SessionState(session_id="s", current_model="fast-9b", last_switch_ts=100.0)
    result = route(
        config,
        decision("general", 2.9),
        session=session,
        now=140.0,  # 40s after the last switch, dwell is 120s
    )

    assert result.switch_required is False
    assert result.blocked_by == "dwell"
    assert "dwell" in result.reason


def test_dwell_expires() -> None:
    config = Config()
    session = SessionState(session_id="s", current_model="fast-9b", last_switch_ts=100.0)
    result = route(config, decision("general", 2.9), session=session, now=500.0)

    assert result.switch_required is True
    assert result.selected is DEEP


def test_a_busy_server_is_never_restarted() -> None:
    config = Config()
    result = route(config, decision("general", 2.9), busy=True)

    assert result.switch_required is False
    assert result.blocked_by == "busy"
    assert "generating" in result.reason


def test_short_non_streaming_calls_never_wait_for_a_load() -> None:
    """cptr generates chat titles with max_tokens=100 and a 15s timeout."""
    config = Config()
    result = route(config, decision("general", 2.9), max_tokens=100, streaming=False)
    # The policy did prefer the bigger model; the guard is what stops it.

    assert result.switch_required is False
    assert result.blocked_by == "utility_call"
    assert result.selected is FAST


def test_a_streaming_request_of_the_same_size_may_switch() -> None:
    config = Config()
    result = route(config, decision("general", 2.9), max_tokens=100, streaming=True)

    assert result.switch_required is True


def test_an_unavailable_executor_downgrades_to_a_recommendation() -> None:
    config = Config()
    result = route(config, decision("general", 2.9), executor_available=False)

    assert result.switch_required is False
    assert result.blocked_by == "executor"
    assert result.would_switch_to is DEEP


def test_shadow_mode_reports_instead_of_switching() -> None:
    config = Config()
    config.routing.mode = "shadow"
    result = route(config, decision("general", 2.9))

    assert result.switch_required is False
    assert result.blocked_by == "mode"
    assert result.would_switch_to is DEEP
    assert "shadow" in result.reason


# --- feasibility ------------------------------------------------------------


def test_a_model_whose_context_is_too_small_is_infeasible() -> None:
    config = Config()
    # 30k tokens of prompt: deep-27b is configured for 20480.
    result = route(config, decision("general", 2.9), estimated_prompt_tokens=30_000)

    assert result.selected is not DEEP
    deep = next(c for c in result.candidates if c.model is DEEP)
    assert deep.feasible is False
    assert "20480" in (deep.infeasible_reason or "")


def test_tools_exclude_a_model_whose_template_cannot_express_them() -> None:
    config = Config()
    result = route(config, decision("medical", 2.9), has_tools=True)

    med = next(c for c in result.candidates if c.model is MED)
    assert med.feasible is False
    assert "tool" in (med.infeasible_reason or "")
    assert result.selected is not MED


def test_medical_work_reaches_the_specialist_without_tools() -> None:
    config = Config()
    result = route(config, decision("medical", 2.2), has_tools=False)

    assert result.selected is MED
    assert result.switch_required is True


def test_ram_guard_warns_by_default_and_blocks_when_configured() -> None:
    config = Config()
    warned = route(config, decision("medical", 2.2), memory=ALMOST_NO_RAM)
    med_candidate = next(c for c in warned.candidates if c.model is MED)
    assert med_candidate.feasible is True
    assert any("host RAM" in warning for warning in med_candidate.warnings)

    config.routing.guards.ram_guard = "block"
    blocked = route(config, decision("medical", 2.2), memory=ALMOST_NO_RAM)
    med_blocked = next(c for c in blocked.candidates if c.model is MED)
    assert med_blocked.feasible is False
    assert "host RAM" in (med_blocked.infeasible_reason or "")
    assert blocked.selected is not MED


def test_nothing_feasible_falls_back_to_the_resident_model() -> None:
    config = Config()
    result = route(
        config,
        decision("general", 2.0),
        estimated_prompt_tokens=10_000_000,
    )

    assert result.selected is FAST  # the resident one, because nothing fits
    assert result.switch_required is False
    assert "No model satisfies" in result.reason


# --- explicit selection -----------------------------------------------------


def test_an_explicit_model_bypasses_automatic_routing() -> None:
    config = Config()
    # A trivial task that would obviously stay on the 9B.
    result = route(config, decision("general", 0.2), explicit_model=MED)

    assert result.explicit is True
    assert result.selected is MED
    assert result.switch_required is True
    assert "named by the request" in result.reason


def test_an_explicit_model_that_is_already_resident_does_not_switch() -> None:
    config = Config()
    result = route(config, decision("general", 0.2), explicit_model=FAST)

    assert result.explicit is True
    assert result.switch_required is False
    assert "already loaded" in result.reason


def test_an_explicit_model_cannot_be_loaded_without_an_executor() -> None:
    config = Config()
    result = route(config, decision("general", 0.2), explicit_model=DEEP, executor_available=False)

    assert result.switch_required is False
    assert result.selected is FAST
    assert result.would_switch_to is DEEP


# --- policy presets ---------------------------------------------------------


def test_fast_policy_never_leaves_the_resident_model() -> None:
    config = Config()
    result = route(config, decision("general", 3.0), policy="fast")

    assert result.selected is FAST
    assert result.switch_required is False
    # Either it never preferred anything else, or the preset refused the move.
    assert result.blocked_by in (None, "policy")


def test_deep_policy_raises_the_difficulty_floor() -> None:
    config = Config()
    result = route(config, decision("general", 0.5), policy="deep")

    assert result.selected is DEEP
    assert result.difficulty_used >= 3.0


def test_medical_policy_routes_by_domain_not_by_words() -> None:
    config = Config()
    # The judgment says "general"; the preset overrides the domain.
    result = route(config, decision("general", 1.5), policy="medical")

    assert result.domain_used == "medical"
    assert result.selected is MED


# --- stickiness -------------------------------------------------------------


def test_a_hard_conversation_does_not_fall_back_on_one_short_turn() -> None:
    """Stickiness: the running difficulty keeps the conversation where it is."""
    config = Config()
    session = SessionState(session_id="s", current_model="deep-27b", difficulty_ewma=2.8)
    result = route(
        config,
        decision("general", 0.3),  # one short follow-up
        resident=DEEP,
        session=session,
        now=10_000.0,
    )

    assert result.selected is DEEP
    assert result.switch_required is False
    assert result.difficulty_used >= 2.8
    # It did not even want to move.
    assert result.would_switch_to is None


def test_alternating_difficulty_does_not_bounce_between_models() -> None:
    """The anti-thrash property, over a conversation rather than one request.

    Six turns alternating hard and easy. A per-turn decision would move the
    model on every one of them; the running difficulty plus the guards allow a
    single upgrade and then hold.
    """
    config = Config()
    policy = RoutingPolicy(config)
    session = SessionState(session_id="s")
    resident = FAST
    switches = 0
    clock = 0.0

    for difficulty in (2.9, 0.4, 2.8, 0.3, 2.9, 0.5):
        clock += 300.0  # well past the dwell window, so dwell is not doing the work
        result = policy.route(
            RouteRequest(streaming=True, session=session, now=clock),
            decision("general", difficulty),
            snapshot(),
            resident=resident,
            memory=PLENTY_OF_RAM,
            server_busy=False,
            executor_available=True,
        )
        selected = result.selected
        assert selected is not None
        if selected.id != resident.id:
            switches += 1
            resident = selected
            session.last_switch_ts = clock
        session.current_model = resident.id
        session.difficulty_ewma = session.updated_difficulty(
            difficulty, config.policy.difficulty_ewma_alpha
        )

    assert switches == 1, "the conversation should settle, not oscillate"
    assert resident is DEEP


# --- honesty about unknowns -------------------------------------------------


def test_an_unmeasured_model_is_never_chosen_for_looking_fast() -> None:
    config = Config()
    unknown = model("unknown-13b", general=2.6, gen_tps=None, load_s=None, size_gb=9.0)
    result = route(
        config,
        decision("general", 2.9),
        models=(FAST, unknown),
        resident=FAST,
    )

    candidate = next(c for c in result.candidates if c.model is unknown)
    # Treated as slow (5 tok/s floor) and as costing a size-derived load.
    assert candidate.generate_s > 100
    assert candidate.switch_s > 0


def test_candidates_expose_their_arithmetic() -> None:
    config = Config()
    result = route(config, decision("coding", 2.4))

    described = {item["model"]: item for item in result.describe()["candidates"]}
    assert set(described) == {"fast-9b", "deep-27b", "med-27b"}
    for item in described.values():
        assert {"quality", "switch_s", "generate_s", "utility", "feasible"} <= set(item)


# --- calibrated distributions -------------------------------------------------


def calibrated(domain_probabilities: dict[str, float], level_probabilities: dict[str, float]) -> TaskDecision:
    """A Jev-shaped decision: calibrated distributions plus their point summary."""
    top = max(domain_probabilities, key=domain_probabilities.get)
    mean = sum(int(level) * p for level, p in level_probabilities.items())
    return TaskDecision(
        domain=top,
        domain_probabilities=domain_probabilities,
        difficulty=mean,
        difficulty_probabilities=level_probabilities,
        backend="jev",
        calibrated=True,
    )


def test_uncalibrated_distributions_are_read_as_point_values() -> None:
    """Rule scores and next-token distributions must not be used as weights."""
    config = Config()
    split = {"general": 0.55, "medical": 0.45}
    levels = {"1": 0.5, "3": 0.5}
    heuristic = TaskDecision(
        domain="general",
        domain_probabilities=split,
        difficulty=2.0,
        difficulty_probabilities=levels,
        backend="rules",
        calibrated=False,
    )
    point = decision("general", 2.0)

    a = route(config, heuristic)
    b = route(config, point)
    assert [c.quality for c in a.candidates] == [c.quality for c in b.candidates]
    assert a.selected is b.selected


def test_a_calibrated_domain_split_leans_toward_the_specialist() -> None:
    """45% medical should pull toward the medical model; the top answer alone would not."""
    config = Config()
    split = calibrated({"general": 0.55, "medical": 0.45}, {"2": 1.0})
    top_only = decision("general", 2.0)

    med_split = next(c for c in route(config, split).candidates if c.model is MED)
    med_top = next(c for c in route(config, top_only).candidates if c.model is MED)
    assert med_split.quality > med_top.quality


def test_levels_are_scored_separately_not_at_their_mean() -> None:
    """Half level 1, half level 3 is not the same task as level 2 on a steep curve."""
    config = Config()
    spread = calibrated({"general": 1.0}, {"1": 0.5, "3": 0.5})
    middle = calibrated({"general": 1.0}, {"2": 1.0})

    fast_spread = next(c for c in route(config, spread).candidates if c.model is FAST)
    fast_middle = next(c for c in route(config, middle).candidates if c.model is FAST)
    assert fast_spread.quality != pytest.approx(fast_middle.quality)


def test_the_reason_names_a_meaningful_domain_split() -> None:
    config = Config()
    result = route(config, calibrated({"general": 0.55, "medical": 0.45}, {"1": 1.0}))
    assert "general 55%" in result.reason and "medical 45%" in result.reason


def test_the_session_floor_applies_to_every_level() -> None:
    config = Config()
    session = SessionState(session_id="s", difficulty_ewma=2.8)
    easy = calibrated({"general": 1.0}, {"0": 0.9, "1": 0.1})

    floored = route(config, easy, resident=DEEP, session=session, now=10_000.0)
    assert floored.selected is DEEP
    assert floored.difficulty_used >= 2.8


@pytest.mark.parametrize(
    ("difficulty", "ceiling", "words"),
    [(2.4, 2.0, "is above"), (2.0, 2.0, "sits at"), (2.04, 2.0, "sits at"), (1.2, 2.0, "sits within")],
)
def test_the_reason_states_difficulty_against_ceiling_as_it_is(
    difficulty: float, ceiling: float, words: str
) -> None:
    """A switch at the ceiling once read "difficulty 2.0 is above ... 2.0 ceiling"."""
    from switchboard.policy import _against_ceiling

    assert _against_ceiling(difficulty, ceiling) == words


def test_context_fit_uses_the_answer_length_the_policy_plans_for() -> None:
    """The deep preset floors difficulty at hard (a ~1200-token answer). The fit
    check used the raw difficulty (an ~80-token answer), so a 27B could be
    chosen for a prompt that left no room for the answer it was chosen for."""
    config = Config()
    result = route(config, decision("general", 0.0), estimated_prompt_tokens=19_500, policy="deep")

    deep = next(c for c in result.candidates if c.model is DEEP)
    assert deep.feasible is False
    assert "20480" in (deep.infeasible_reason or "")


def test_ceiling_words_always_agree_with_the_printed_numbers() -> None:
    """Half-steps such as an EWMA of 1.85 used to print "1.9 sits at its 1.8"."""
    from switchboard.policy import _against_ceiling

    for tenths in range(0, 400):
        difficulty = tenths / 100 + 0.005  # every x.xx5 half-step, the risky ones
        for ceiling in (1.0, 1.6, 1.8, 2.0, 2.5):
            shown_d, shown_c = f"{difficulty:.1f}", f"{ceiling:.1f}"
            words = _against_ceiling(difficulty, ceiling)
            expected = (
                "is above" if float(shown_d) > float(shown_c)
                else "sits at" if shown_d == shown_c
                else "sits within"
            )
            assert words == expected, (difficulty, ceiling, words)


# --- models turned off for automatic switching --------------------------------


def test_a_model_turned_off_for_automatic_switching_is_never_chosen() -> None:
    config = Config()
    config.routing.auto_exclude = ["deep-27b"]

    result = route(config, decision("general", 2.9))

    deep = next(c for c in result.candidates if c.model is DEEP)
    assert deep.feasible is False and "turned off in Settings" in (deep.infeasible_reason or "")
    assert result.selected is not DEEP


def test_an_excluded_model_that_is_loaded_keeps_serving() -> None:
    config = Config()
    config.routing.auto_exclude = ["med-27b"]

    result = route(config, decision("medical", 1.5), resident=MED)

    assert result.selected is MED and result.switch_required is False


def test_an_excluded_model_can_still_be_requested_by_name() -> None:
    config = Config()
    config.routing.auto_exclude = ["deep-27b"]

    result = route(config, decision("general", 1.0), explicit_model=DEEP)

    assert result.selected is DEEP and result.switch_allowed is True
