"""Whether a Lecture Notes recording is running right now.

A lecture being transcribed shares the GPU with llama-server. Loading a large
model in the middle of it can take the memory the transcription model needs, so
the routing policy holds large switches back while one is running (see the
recording guard in `policy.py`).

Lecture Notes already publishes its state: the recorder rewrites
``state/status.json`` atomically every five seconds while a session is active,
and sets ``"active": false`` when it ends. Switchboard only reads that file; it
never writes to it or talks to the recorder.

Three cases read as "not recording", so a problem on the Lecture Notes side can
never block switching forever: the file is missing or unreadable, it says
``active: false``, or it has not been rewritten for `STALE_AFTER_S` seconds
(twelve missed heartbeats means the recorder is gone, not busy).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

# The recorder's heartbeat is five seconds; a minute without one is a dead recorder.
STALE_AFTER_S = 60.0
# How long one read of the file is reused; routing asks on every request.
CACHE_S = 2.0


@dataclass(frozen=True, slots=True)
class RecordingState:
    active: bool
    # What is being recorded, e.g. "CHEM 1450", when the recorder says.
    label: str | None = None
    # Why the state is what it is, for doctor and the dashboard.
    detail: str = ""

    def describe(self) -> dict[str, Any]:
        return {"active": self.active, "label": self.label, "detail": self.detail}


NOT_RECORDING = RecordingState(active=False, detail="no recording")


def read_recording_state(path: Path, *, now: float | None = None) -> RecordingState:
    """Read the recorder's status file once."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return RecordingState(active=False, detail="Lecture Notes has no status file")
    except (OSError, ValueError):
        return RecordingState(active=False, detail="Lecture Notes status file is unreadable")
    if not isinstance(data, dict) or data.get("active") is not True:
        return RecordingState(active=False, detail="Lecture Notes is not recording")

    label = data.get("class_code")
    label = label if isinstance(label, str) and label.strip() else None
    updated = data.get("updated_at")
    current = time.time() if now is None else now
    if not isinstance(updated, (int, float)) or current - updated > STALE_AFTER_S:
        return RecordingState(
            active=False,
            label=label,
            detail="Lecture Notes says it is recording but has not updated its status "
            f"for over {STALE_AFTER_S:.0f}s; treated as stopped",
        )
    started = data.get("started_at")
    minutes = (current - started) / 60 if isinstance(started, (int, float)) else None
    detail = f"recording {label or 'a lecture'}"
    if minutes is not None and minutes >= 0:
        detail += f", started {minutes:.0f} min ago"
    return RecordingState(active=True, label=label, detail=detail)


class RecordingWatch:
    """Cached view of the recorder's status, cheap enough to ask on every request."""

    def __init__(self, path: Path, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._path = path
        self._clock = clock
        self._cached: RecordingState | None = None
        self._read_at = 0.0

    @property
    def path(self) -> Path:
        return self._path

    def state(self) -> RecordingState:
        now = self._clock()
        if self._cached is None or now - self._read_at >= CACHE_S:
            self._cached = read_recording_state(self._path)
            self._read_at = now
        return self._cached
