"""Tests for the observability store.

The store is the only thing that survives a restart, so these tests care most
about two things: that an existing database on disk keeps working after an
upgrade, and that the privacy contract in the module docstring is enforced by
the code rather than by the good manners of its callers.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import UTC, datetime, timedelta
from inspect import signature
from pathlib import Path

import pytest

from switchboard.registry import Measurement
from switchboard.store import (
    EWMA_ALPHA,
    EWMA_WINDOW,
    MAX_HASH_CHARS,
    SCHEMA_VERSION,
    Store,
    ewma,
    iso_utc,
    percentile,
)


@pytest.fixture
def store() -> Store:
    with Store(Path(":memory:")) as opened:
        yield opened


def days_ago(days: float) -> str:
    return iso_utc(datetime.now(UTC) - timedelta(days=days))


def hours_ago(hours: float) -> str:
    return iso_utc(datetime.now(UTC) - timedelta(hours=hours))


def table_columns(store: Store) -> dict[str, list[str]]:
    """Every user table and its columns, read back out of SQLite itself."""
    names = [
        row["name"]
        for row in store._query(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name NOT LIKE 'sqlite_%'"
        )
    ]
    return {
        name: [row["name"] for row in store._query(f'PRAGMA table_info("{name}")')]
        for name in names
    }


# --- schema and migrations --------------------------------------------------


def test_schema_is_created_with_every_table(store: Store) -> None:
    columns = table_columns(store)
    assert set(columns) == {
        "routes",
        "decision_attempts",
        "switches",
        "step_evals",
        "measurements",
        "sessions",
    }
    assert store.schema_version == SCHEMA_VERSION


def test_pragmas_are_set(tmp_path: Path) -> None:
    # WAL is only meaningful for a file database; :memory: reports "memory".
    with Store(tmp_path / "sb.sqlite3") as store:
        assert store._query("PRAGMA journal_mode")[0][0] == "wal"
        assert store._query("PRAGMA synchronous")[0][0] == 1  # NORMAL
        assert store._query("PRAGMA foreign_keys")[0][0] == 1


def test_constructor_creates_parent_directories(tmp_path: Path) -> None:
    path = tmp_path / "data" / "nested" / "sb.sqlite3"
    with Store(path):
        pass
    assert path.is_file()


def test_reopening_is_idempotent_and_keeps_data(tmp_path: Path) -> None:
    path = tmp_path / "sb.sqlite3"
    with Store(path) as first:
        route_id = first.record_route(source="proxy", selected_model="m")
        before = table_columns(first)

    with Store(path) as second:
        assert second.schema_version == SCHEMA_VERSION
        assert table_columns(second) == before
        assert second.get_route(route_id) is not None


def test_older_database_upgrades_in_place(tmp_path: Path) -> None:
    """A database written before the version stamp must upgrade, not break."""
    path = tmp_path / "sb.sqlite3"
    with Store(path) as first:
        route_id = first.record_route(source="cli", reason="seed")

    # Rewind the stamp the way a pre-migration database on disk would look.
    raw = sqlite3.connect(path)
    raw.execute("PRAGMA user_version = 0")
    raw.commit()
    raw.close()

    with Store(path) as upgraded:
        assert upgraded.schema_version == SCHEMA_VERSION
        # Migrating must not have dropped what was already there.
        route = upgraded.get_route(route_id)
        assert route is not None and route["reason"] == "seed"
        assert upgraded.record_route(source="cli") > route_id


# --- routes -----------------------------------------------------------------


def test_route_record_and_update_round_trip(store: Store) -> None:
    route_id = store.record_route(
        source="proxy",
        session_id="sess-1",
        mode="active",
        policy="auto",
        requested_model=None,
        selected_model="qwen3.5-9b",
        resident_before="medgemma-27b",
        switched=True,
        switch_ms=27600.0,
        decision_backend="jev",
        decision_latency_ms=310.5,
        decision={"domain": "coding", "difficulty": 2.4},
        reason="coding task beyond resident capability",
        confidence=0.81,
        prompt_hash="a1b2c3d4e5f60718",
        preview="refactor the retry loop",
        estimated_prompt_tokens=412,
        streaming=True,
    )
    assert isinstance(route_id, int) and route_id > 0

    assert store.update_route(
        route_id,
        downstream_status=200,
        ttfb_ms=640.0,
        downstream_ms=9100.0,
        tokens_in=412,
        tokens_out=730,
        gen_tps=18.4,
        error=None,
        outcome="ok",
    )

    route = store.get_route(route_id)
    assert route is not None
    assert route["session_id"] == "sess-1"
    assert route["source"] == "proxy"
    assert route["selected_model"] == "qwen3.5-9b"
    assert route["resident_before"] == "medgemma-27b"
    assert route["prompt_hash"] == "a1b2c3d4e5f60718"
    assert route["preview"] == "refactor the retry loop"
    assert route["estimated_prompt_tokens"] == 412
    assert route["downstream_status"] == 200
    assert route["ttfb_ms"] == 640.0
    assert route["downstream_ms"] == 9100.0
    assert route["tokens_out"] == 730
    assert route["gen_tps"] == 18.4
    assert route["outcome"] == "ok"
    # 0/1 comes back as bool, JSON text comes back parsed.
    assert route["switched"] is True
    assert route["streaming"] is True
    assert route["decision"] == {"domain": "coding", "difficulty": 2.4}


def test_update_route_rejects_columns_it_does_not_own(store: Store) -> None:
    route_id = store.record_route(source="api")
    with pytest.raises(ValueError, match="cannot set"):
        store.update_route(route_id, prompt_hash="deadbeef")
    with pytest.raises(ValueError, match="cannot set"):
        store.update_route(route_id, preview="a prompt sneaking in the back door")
    # The rejected call must not have written anything either.
    assert store.get_route(route_id)["preview"] is None


def test_update_route_reports_a_missing_row(store: Store) -> None:
    assert store.update_route(99999, outcome="ok") is False
    assert store.update_route(store.record_route(source="api")) is False


def test_unknown_route_source_is_rejected(store: Store) -> None:
    with pytest.raises(ValueError, match="source must be one of"):
        store.record_route(source="telepathy")  # type: ignore[arg-type]


def test_recent_routes_are_newest_first(store: Store) -> None:
    ids = [store.record_route(source="proxy", reason=f"r{index}") for index in range(5)]
    recent = store.recent_routes(limit=3)
    assert [row["id"] for row in recent] == list(reversed(ids))[:3]
    assert len(store.recent_routes(limit=50)) == 5


# --- privacy ----------------------------------------------------------------

# Names a column must never have: each could hold a prompt, a response, or an
# agent transcript. `prompt_hash`/`goal_hash` are fine, `prompt` is not.
FORBIDDEN_COLUMNS = frozenset(
    {
        "prompt",
        "prompt_text",
        "prompts",
        "message",
        "messages",
        "text",
        "content",
        "body",
        "request",
        "request_body",
        "response",
        "response_body",
        "completion",
        "transcript",
        "system_prompt",
        "goal",
        "action",
        "result",
        "user_input",
        "raw",
    }
)


def test_no_table_has_a_column_that_could_hold_a_raw_prompt(store: Store) -> None:
    for table, columns in table_columns(store).items():
        for column in columns:
            assert column not in FORBIDDEN_COLUMNS, f"{table}.{column} could hold a prompt"
            # Anything that mentions a prompt must be a hash or a token count.
            if "prompt" in column:
                assert column.endswith(("_hash", "_tokens")), f"{table}.{column}"


def test_no_write_method_accepts_prompt_text(store: Store) -> None:
    writers = (
        Store.record_route,
        Store.record_attempt,
        Store.record_switch,
        Store.record_step_eval,
        Store.record_measurement,
        Store.upsert_session,
    )
    for method in writers:
        for name in signature(method).parameters:
            assert name not in FORBIDDEN_COLUMNS, f"{method.__name__}({name}=...)"


def test_preview_is_redacted_and_truncated_by_the_store(tmp_path: Path) -> None:
    """The store does not trust the caller to have shortened or cleaned it."""
    with Store(tmp_path / "sb.sqlite3", preview_chars=40) as store:
        secret = "sk-abcdefghijklmnopqrstuvwxyz012345"
        essay = "please review\nthis " + ("very long prompt body " * 20)
        route_id = store.record_route(
            source="proxy", preview=f"{essay} token={secret}"
        )
        stored = store.get_route(route_id)["preview"]

    assert len(stored) <= 40
    assert secret not in stored
    assert "\n" not in stored
    # Only the head of the request survives; the body does not.
    assert stored.startswith("please review this")
    assert stored.count("very long prompt body") <= 1


def test_preview_chars_cannot_be_widened_into_a_prompt_column(tmp_path: Path) -> None:
    with Store(tmp_path / "sb.sqlite3", preview_chars=100_000) as store:
        route_id = store.record_route(source="proxy", preview="x" * 5_000)
        assert len(store.get_route(route_id)["preview"]) <= 200


def test_prompt_text_is_never_accepted_in_place_of_a_hash(store: Store) -> None:
    prompt = "Write me a function that parses GGUF headers " * 10
    with pytest.raises(ValueError, match="must be a hash"):
        store.record_route(source="proxy", prompt_hash=prompt)
    with pytest.raises(ValueError, match="must be a hash"):
        store.record_step_eval(goal_hash="ship the feature\nwithout breaking tests")
    with pytest.raises(ValueError, match="must be a hash"):
        store.record_step_eval(action_hash="a" * (MAX_HASH_CHARS + 1))
    # Nothing was written by any of the rejected calls.
    assert store.recent_routes() == []
    assert store.recent_step_evals() == []


# --- decision attempts and step evaluations ---------------------------------


def test_attempt_may_hang_off_a_route_or_stand_alone(store: Store) -> None:
    route_id = store.record_route(source="proxy")
    store.record_attempt(
        backend="jev",
        kind="task",
        ok=True,
        route_id=route_id,
        latency_ms=290.0,
        input_tokens=180,
        est_cost_usd=0.0000076,
    )
    store.record_attempt(backend="local", kind="step", ok=False, error_kind="timeout")

    stats = store.provider_stats(hours=24)
    assert stats["jev"]["attempts"] == 1 and stats["jev"]["ok"] == 1
    assert stats["local"]["ok"] == 0
    assert stats["local"]["error_kinds"] == {"timeout": 1}

    with pytest.raises(ValueError, match="kind must be one of"):
        store.record_attempt(backend="jev", kind="vibes", ok=True)  # type: ignore[arg-type]


def test_step_eval_round_trip(store: Store) -> None:
    store.record_step_eval(
        session_id="sess-2",
        goal_hash="0badc0de0badc0de",
        action_hash="feedfacefeedface",
        attempt=2,
        outcome="partial",
        progress=0.6,
        complete=0.1,
        needs_new_approach=0.8,
        recommendation="retry_different",
        backend="local",
        latency_ms=1450.0,
    )
    row = store.recent_step_evals()[0]
    assert row["outcome"] == "partial"
    assert row["attempt"] == 2
    assert row["recommendation"] == "retry_different"
    assert row["goal_hash"] == "0badc0de0badc0de"


# --- switches ---------------------------------------------------------------


def test_switches_round_trip_and_default_to_unconfirmed(store: Store) -> None:
    store.record_switch(from_model="a", to_model="b", executor="dashboard")
    store.record_switch(
        from_model="b",
        to_model="c",
        executor="dashboard",
        trigger="explicit",
        ok=True,
        load_ms=27600.0,
    )
    newest, oldest = store.recent_switches(limit=10)
    assert newest["to_model"] == "c" and newest["ok"] is True
    assert newest["trigger"] == "explicit" and newest["load_ms"] == 27600.0
    # A caller that never confirms leaves a visible failure rather than a success.
    assert oldest["ok"] is False and oldest["trigger"] == "auto"

    with pytest.raises(ValueError, match="trigger must be one of"):
        store.record_switch(from_model="a", to_model="b", trigger="whim")  # type: ignore[arg-type]


# --- measurements and the weighted mean -------------------------------------


def test_ewma_follows_the_documented_recurrence() -> None:
    assert ewma([]) is None
    assert ewma([12.0]) == 12.0
    # Seeded with the first sample, then alpha * new + (1 - alpha) * running.
    assert EWMA_ALPHA == pytest.approx(0.3)
    assert ewma([10.0, 20.0, 30.0]) == pytest.approx(18.1)
    assert ewma([10.0, 20.0, 30.0], alpha=0.5) == pytest.approx(22.5)
    # The weights sum to one, so the mean cannot leave the range of its samples.
    assert 4.0 <= ewma([4.0, 9.0, 5.0, 7.0]) <= 9.0


def test_measurements_by_model_weights_recent_samples(store: Store) -> None:
    for index, value in enumerate((10.0, 20.0, 30.0)):
        store.record_measurement(
            model_id="qwen3.5-9b",
            metric="gen_tps",
            value=value,
            source="proxy",
            context={"tokens": 100 + index},
            ts=days_ago(3 - index),
        )
    store.record_measurement(model_id="qwen3.5-9b", metric="load_s", value=27.6)
    store.record_measurement(model_id="medgemma-27b", metric="gen_tps", value=5.0)

    measured = store.measurements_by_model()
    gen = measured["qwen3.5-9b"]["gen_tps"]
    assert isinstance(gen, Measurement)
    assert gen.value == pytest.approx(18.1)
    assert gen.samples == 3
    assert gen.measured is True
    assert gen.source.startswith("measured on this machine (n=3, latest ")
    # The date named is the newest sample's, which here is yesterday's.
    assert gen.source.endswith(f"{days_ago(1)[:10]})")

    assert measured["qwen3.5-9b"]["load_s"].value == pytest.approx(27.6)
    assert measured["medgemma-27b"]["gen_tps"].samples == 1


def test_measurements_fold_only_the_most_recent_window(store: Store) -> None:
    # An early outlier beyond the window must not reach the answer at all.
    store.record_measurement(
        model_id="m", metric="gen_tps", value=10_000.0, ts=days_ago(900)
    )
    for index in range(EWMA_WINDOW):
        store.record_measurement(
            model_id="m", metric="gen_tps", value=20.0, ts=days_ago(800 - index)
        )
    folded = store.measurements_by_model()["m"]["gen_tps"]
    assert folded.samples == EWMA_WINDOW
    assert folded.value == pytest.approx(20.0)


def test_measurement_metric_is_constrained(store: Store) -> None:
    with pytest.raises(ValueError, match="metric must be one of"):
        store.record_measurement(model_id="m", metric="vram_gb", value=1.0)  # type: ignore[arg-type]


def test_measurement_context_round_trips_as_json(store: Store) -> None:
    store.record_measurement(
        model_id="m", metric="load_s", value=3.0, context={"gpu_layers": 48}
    )
    row = store._query("SELECT * FROM measurements")[0]
    assert row["context"] == '{"gpu_layers":48}'


# --- percentiles ------------------------------------------------------------


def test_percentile_uses_nearest_rank() -> None:
    assert percentile([], 0.5) is None
    assert percentile([7.0], 0.95) == 7.0
    values = [float(n) for n in range(1, 101)]  # 1..100
    assert percentile(values, 0.50) == 50.0
    assert percentile(values, 0.95) == 95.0
    assert percentile(values, 1.0) == 100.0
    # Unsorted input, and a rank that must clamp rather than run off the end.
    assert percentile([30.0, 10.0, 20.0], 0.5) == 20.0
    assert percentile([30.0, 10.0, 20.0], 0.0) == 10.0
    # Nearest rank returns an observed value, never an interpolated one.
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.0


def test_provider_stats_percentiles_and_cost(store: Store) -> None:
    for latency in range(1, 21):  # 1..20 ms
        store.record_attempt(
            backend="jev",
            kind="task",
            ok=True,
            latency_ms=float(latency),
            est_cost_usd=0.001,
        )
    store.record_attempt(backend="jev", kind="task", ok=False, error_kind="timeout")
    store.record_attempt(backend="jev", kind="task", ok=False, error_kind="timeout")
    store.record_attempt(backend="jev", kind="task", ok=False)

    jev = store.provider_stats(hours=24)["jev"]
    assert jev["attempts"] == 23
    assert jev["ok"] == 20
    assert jev["error_kinds"] == {"timeout": 2, "unknown": 1}
    assert jev["p50_latency_ms"] == 10.0
    assert jev["p95_latency_ms"] == 19.0
    assert jev["est_cost_usd"] == pytest.approx(0.02)


def test_provider_stats_respects_the_window(store: Store) -> None:
    store.record_attempt(backend="jev", kind="task", ok=True, ts=hours_ago(48))
    store.record_attempt(backend="jev", kind="task", ok=True, ts=hours_ago(1))
    assert store.provider_stats(hours=24)["jev"]["attempts"] == 1
    assert store.provider_stats(hours=72)["jev"]["attempts"] == 2
    # Zero or negative means "everything ever recorded".
    assert store.provider_stats(hours=0)["jev"]["attempts"] == 2


def test_route_stats_summarises_the_window(store: Store) -> None:
    fell_back = store.record_route(
        source="proxy",
        selected_model="qwen3.5-9b",
        decision_backend="local",
        switched=True,
        streaming=True,
        decision_latency_ms=100.0,
        reason="jev timed out",
    )
    # Two failures on one route: a fallback is a route, not an attempt count.
    store.record_attempt(backend="jev", kind="task", ok=False, error_kind="timeout",
                         route_id=fell_back)
    store.record_attempt(backend="jev", kind="task", ok=False, error_kind="timeout",
                         route_id=fell_back)
    store.record_attempt(backend="local", kind="task", ok=True, route_id=fell_back)
    store.update_route(fell_back, downstream_ms=5000.0)

    clean = store.record_route(
        source="proxy",
        selected_model="qwen3.5-9b",
        decision_backend="jev",
        streaming=False,
        decision_latency_ms=300.0,
    )
    store.record_attempt(backend="jev", kind="task", ok=True, route_id=clean)
    store.update_route(clean, downstream_ms=1000.0)

    store.record_route(source="shadow", decision_backend="rules", streaming=False)
    store.record_route(source="proxy", selected_model="medgemma-27b", ts=hours_ago(48))

    stats = store.route_stats(hours=24)
    assert stats["routes"] == 3
    assert stats["by_model"] == {"qwen3.5-9b": 2, "unrouted": 1}
    assert stats["by_backend"] == {"local": 1, "jev": 1, "rules": 1}
    assert stats["switches"] == 1
    # One route had a backend fail before another answered; the clean one did not.
    assert stats["fallbacks"] == 1
    assert stats["p50_decision_ms"] == 100.0
    assert stats["p95_decision_ms"] == 300.0
    assert stats["p50_downstream_ms"] == 1000.0
    assert stats["p95_downstream_ms"] == 5000.0
    assert stats["streaming_share"] == pytest.approx(1 / 3)

    # The older route is outside the 24h window but inside a wider one.
    assert store.route_stats(hours=72)["by_model"]["medgemma-27b"] == 1


def test_route_stats_on_an_empty_store(store: Store) -> None:
    stats = store.route_stats()
    assert stats["routes"] == 0
    assert stats["streaming_share"] == 0.0
    assert stats["p50_decision_ms"] is None
    assert store.provider_stats() == {}


# --- sessions ---------------------------------------------------------------


def test_get_session_is_none_before_anything_is_written(store: Store) -> None:
    assert store.get_session("nobody") is None


def test_upsert_session_merges_rather_than_replaces(store: Store) -> None:
    created = store.upsert_session(
        "sess-3", current_model="qwen3.5-9b", difficulty_ewma=1.5, bump_turns=True
    )
    assert created["current_model"] == "qwen3.5-9b"
    assert created["difficulty_ewma"] == 1.5
    assert created["turns"] == 1
    assert created["created_ts"] and created["updated_ts"]

    # A later call that mentions only one field must leave the others standing.
    merged = store.upsert_session("sess-3", difficulty_ewma=2.25, bump_turns=True)
    assert merged["current_model"] == "qwen3.5-9b"
    assert merged["difficulty_ewma"] == 2.25
    assert merged["turns"] == 2
    assert merged["created_ts"] == created["created_ts"]

    # And a call that mentions nothing at all is still a valid touch.
    touched = store.upsert_session("sess-3")
    assert touched["turns"] == 2
    assert touched["current_model"] == "qwen3.5-9b"

    switched = store.upsert_session(
        "sess-3", current_model="medgemma-27b", last_switch_ts=iso_utc()
    )
    assert switched["current_model"] == "medgemma-27b"
    assert switched["difficulty_ewma"] == 2.25
    assert store.get_session("sess-3") == switched


def test_upsert_session_rejects_unknown_fields(store: Store) -> None:
    with pytest.raises(ValueError, match="cannot set"):
        store.upsert_session("sess-4", created_ts="1999-01-01T00:00:00.000Z")
    with pytest.raises(ValueError, match="cannot set"):
        store.upsert_session("sess-4", last_message="hello there")
    assert store.get_session("sess-4") is None


def test_turn_counter_survives_concurrent_writers(store: Store) -> None:
    """The counter is incremented in SQL, so threads cannot lose an update."""

    def bump() -> None:
        for _ in range(20):
            store.upsert_session("busy", bump_turns=True)

    threads = [threading.Thread(target=bump) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert store.get_session("busy")["turns"] == 80


def test_writes_from_many_threads_all_land(store: Store) -> None:
    def write() -> None:
        for _ in range(25):
            route_id = store.record_route(source="proxy", selected_model="m")
            store.record_attempt(backend="rules", kind="task", ok=True, route_id=route_id)
            store.update_route(route_id, outcome="ok")

    threads = [threading.Thread(target=write) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    routes = store.recent_routes(limit=1000)
    assert len(routes) == 200
    assert len({row["id"] for row in routes}) == 200
    assert all(row["outcome"] == "ok" for row in routes)
    assert store.provider_stats(hours=0)["rules"]["attempts"] == 200


# --- maintenance ------------------------------------------------------------


def test_prune_drops_old_rows_and_keeps_recent_ones(tmp_path: Path) -> None:
    path = tmp_path / "sb.sqlite3"
    store = Store(path)

    old_route = store.record_route(source="proxy", ts=days_ago(60))
    store.record_attempt(backend="jev", kind="task", ok=False, route_id=old_route,
                         ts=days_ago(60))
    store.record_switch(from_model="a", to_model="b", ts=days_ago(60))
    store.record_step_eval(session_id="old", goal_hash="0bad", ts=days_ago(60))
    store.record_measurement(model_id="m", metric="load_s", value=27.6, ts=days_ago(60))
    store.upsert_session("old")

    new_route = store.record_route(source="proxy")
    store.record_attempt(backend="jev", kind="task", ok=True, route_id=new_route)
    store.record_switch(from_model="b", to_model="c")
    store.record_step_eval(session_id="new", goal_hash="feed")
    store.upsert_session("new")
    store.close()

    # upsert_session always stamps `now`, so age the row the way time would.
    raw = sqlite3.connect(path)
    raw.execute(
        "UPDATE sessions SET updated_ts = ? WHERE session_id = 'old'", (days_ago(60),)
    )
    raw.commit()
    raw.close()

    with Store(path) as store:
        deleted = store.prune(retention_days=30)
        assert deleted == {
            "decision_attempts": 1,
            "routes": 1,
            "switches": 1,
            "step_evals": 1,
            "sessions": 1,
        }

        assert [row["id"] for row in store.recent_routes()] == [new_route]
        assert len(store.recent_switches()) == 1
        assert len(store.recent_step_evals()) == 1
        assert store.get_session("old") is None
        assert store.get_session("new") is not None
        assert store.provider_stats(hours=0)["jev"]["attempts"] == 1

        # Measurements are the record of how this machine performs; they stay.
        assert "measurements" not in deleted
        assert store.measurements_by_model()["m"]["load_s"].value == pytest.approx(27.6)

        # Nothing left to do on a second pass.
        assert store.prune(retention_days=30) == {
            "decision_attempts": 0,
            "routes": 0,
            "switches": 0,
            "step_evals": 0,
            "sessions": 0,
        }


def test_prune_also_takes_attempts_belonging_to_a_pruned_route(store: Store) -> None:
    """An attempt younger than the window still goes if its route does."""
    old_route = store.record_route(source="proxy", ts=days_ago(60))
    store.record_attempt(backend="jev", kind="task", ok=True, route_id=old_route)

    deleted = store.prune(retention_days=30)
    assert deleted["routes"] == 1
    assert deleted["decision_attempts"] == 1
    assert store.provider_stats(hours=0) == {}


def test_prune_requires_a_positive_window(store: Store) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        store.prune(retention_days=0)
    with pytest.raises(ValueError, match="must be positive"):
        store.prune(retention_days=-1)


def test_close_is_idempotent(tmp_path: Path) -> None:
    store = Store(tmp_path / "sb.sqlite3")
    store.close()
    store.close()
    with pytest.raises(sqlite3.ProgrammingError):
        store.recent_routes()


def test_vacuum_runs_after_a_prune(tmp_path: Path) -> None:
    with Store(tmp_path / "sb.sqlite3") as store:
        for _ in range(50):
            store.record_route(source="proxy", ts=days_ago(60))
        store.prune(retention_days=30)
        store.vacuum()
        assert store.recent_routes() == []


def test_iso_utc_sorts_chronologically_as_text() -> None:
    moments = [
        iso_utc(datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)),
        iso_utc(datetime(2026, 1, 2, 3, 4, 6, tzinfo=UTC)),
        iso_utc(datetime(2026, 11, 2, 3, 4, 5, tzinfo=UTC)),
    ]
    assert moments == sorted(moments)
    assert all(stamp.endswith("Z") and len(stamp) == len(moments[0]) for stamp in moments)
    # A naive-looking local time is normalised to UTC rather than stored as-is.
    assert iso_utc(datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)) == "2026-01-02T03:04:05.000Z"
