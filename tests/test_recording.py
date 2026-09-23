"""The recording guard: no large model loads while Lecture Notes is recording.

Lecture Notes rewrites its status file every five seconds while a session is
live. A large model loaded mid-lecture could take the GPU memory the
transcription needs, so while that file says "recording", switches to models
over the size limit are held back - and nothing else changes.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest
from conftest import FakeExecutor, FakeLlama
from test_policy import DEEP, FAST, MED, decision, route
from test_proxy import Harness, chat_body

from switchboard.config import Config
from switchboard.recording import (
    CACHE_S,
    STALE_AFTER_S,
    RecordingState,
    RecordingWatch,
    read_recording_state,
)

LIVE = RecordingState(active=True, label="CHEM 1450", detail="recording CHEM 1450")


def write_status(path: Path, *, active: bool = True, age_s: float = 1.0, label: str = "CHEM 1450") -> None:
    now = time.time()
    path.write_text(
        json.dumps({"active": active, "class_code": label, "updated_at": now - age_s, "started_at": now - 600}),
        encoding="utf-8",
    )


# --- reading the recorder's status ---------------------------------------------------


def test_a_fresh_active_status_is_a_recording(tmp_path: Path) -> None:
    path = tmp_path / "status.json"
    write_status(path)

    state = read_recording_state(path)

    assert state.active is True and state.label == "CHEM 1450"
    assert "10 min" in state.detail


@pytest.mark.parametrize(
    ("setup", "why"),
    [
        (lambda p: None, "no status file"),
        (lambda p: p.write_text("{not json", encoding="utf-8"), "unreadable"),
        (lambda p: p.write_text("[1, 2]", encoding="utf-8"), "not recording"),
        (lambda p: write_status(p, active=False), "not recording"),
        (lambda p: write_status(p, age_s=STALE_AFTER_S + 5), "treated as stopped"),
    ],
    ids=["missing", "garbage", "not-an-object", "inactive", "stale"],
)
def test_anything_but_a_fresh_active_status_is_not_a_recording(tmp_path: Path, setup, why: str) -> None:
    """A crashed recorder or a broken file must never hold switches back forever."""
    path = tmp_path / "status.json"
    setup(path)

    state = read_recording_state(path)

    assert state.active is False
    assert why in state.detail


def test_the_watch_rereads_only_after_its_cache_expires(tmp_path: Path) -> None:
    path = tmp_path / "status.json"
    write_status(path)
    clock = [100.0]
    watch = RecordingWatch(path, clock=lambda: clock[0])
    assert watch.state().active is True

    write_status(path, active=False)
    assert watch.state().active is True  # still cached
    clock[0] += CACHE_S
    assert watch.state().active is False


# --- the policy ---------------------------------------------------------------------


def test_a_large_switch_is_held_while_recording() -> None:
    free = route(Config(), decision("general", 2.9))
    assert free.switch_required and free.selected is DEEP  # what it would do otherwise

    held = route(Config(), decision("general", 2.9), recording=LIVE)

    assert held.switch_required is False
    assert held.selected is FAST
    assert held.blocked_by == "recording"
    assert held.would_switch_to is DEEP
    assert "Lecture Notes records CHEM 1450" in held.reason and "9 GB" in held.reason


def test_nothing_is_held_when_no_recording_is_running() -> None:
    idle = RecordingState(active=False, detail="Lecture Notes is not recording")
    assert route(Config(), decision("general", 2.9), recording=idle).selected is DEEP


def test_the_guard_can_be_turned_off_or_the_limit_raised() -> None:
    off = Config()
    off.routing.guards.recording_guard = False
    assert route(off, decision("general", 2.9), recording=LIVE).selected is DEEP

    roomier = Config()
    roomier.routing.guards.recording_max_model_gb = 12.0  # DEEP is 11.9 GB
    assert route(roomier, decision("general", 2.9), recording=LIVE).selected is DEEP


def test_a_smaller_model_is_still_an_option_while_recording() -> None:
    """With a 27B loaded, moving to the 9B can only give memory back."""
    free = route(Config(), decision("coding", 1.5), resident=MED)
    assert free.selected is DEEP, "the unguarded choice for this task"

    held = route(Config(), decision("coding", 1.5), resident=MED, recording=LIVE)

    assert held.selected is FAST
    assert held.switch_required is True
    assert "deep-27b (11.9 GB) would be the first choice" in held.reason


def test_the_loaded_model_is_never_forced_out() -> None:
    """A large model that is already resident keeps serving; only loads are held."""
    stay = route(Config(), decision("medical", 1.5), resident=MED, recording=LIVE)

    assert stay.selected is MED and stay.switch_required is False and stay.blocked_by is None


def test_an_explicit_large_model_is_refused_while_recording() -> None:
    held = route(Config(), decision("general", 1.0), explicit_model=DEEP, recording=LIVE)

    assert held.switch_allowed is False
    assert held.blocked_by == "recording"
    assert held.would_switch_to is DEEP
    assert "held off while Lecture Notes records" in held.reason


def test_an_explicit_small_model_is_still_loaded_while_recording() -> None:
    allowed = route(Config(), decision("general", 1.0), resident=MED, explicit_model=FAST, recording=LIVE)
    assert allowed.switch_allowed is True and allowed.selected is FAST


# --- through the service ---------------------------------------------------------------


@pytest.fixture
async def recording_harness(config: Config) -> Harness:
    write_status(config.stack.recording_status_file)
    # The fake stack's GGUF files are a few kilobytes, so any non-resident model
    # counts as "large" only against a tiny limit.
    config.routing.guards.recording_max_model_gb = 1e-9
    async with Harness(config, FakeLlama(), FakeExecutor()) as harness:
        yield harness


async def test_a_request_naming_a_large_model_gets_a_409_that_says_why(recording_harness: Harness) -> None:
    response = await recording_harness.client.post("/v1/chat/completions", json=chat_body(model="deep-27b"))

    assert response.status_code == 409
    message = response.json()["error"]["message"]
    assert "Lecture Notes" in message and "deep-27b" in message
    assert recording_harness.executor.calls == [], "nothing was loaded"


async def test_a_manual_switch_to_a_large_model_is_refused_while_recording(recording_harness: Harness) -> None:
    deep = recording_harness.router.registry().by_id("deep-27b")

    result = await asyncio.wait_for(recording_harness.router.switch_to(deep), timeout=5)

    assert result.ok is False and "Lecture Notes" in (result.error or "")
    assert recording_harness.executor.calls == []


async def test_status_reports_the_recording_and_the_guard(recording_harness: Harness) -> None:
    status = (await recording_harness.client.get("/status")).json()

    assert status["recording"]["active"] is True
    assert status["recording"]["label"] == "CHEM 1450"
    assert status["recording"]["guard"] is True
    assert status["recording"]["max_model_gb"] == recording_harness.config.routing.guards.recording_max_model_gb
