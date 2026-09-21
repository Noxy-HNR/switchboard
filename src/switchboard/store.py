"""The observability store: the only thing Switchboard remembers.

A router that cannot be audited is a router nobody should trust. Every routing
decision, every backend attempt, every model switch and every measured number
lands here so that "why did it pick that model?" and "is the cloud backend
actually worth its latency?" have answers from this machine rather than from
intuition. The registry also reads its runtime numbers back out of here, which
is how a seeded estimate becomes a measurement.

The rules this module follows:

* **No raw prompts, ever.** Callers identify a request by a hash from
  `switchboard.privacy` plus, at most, a short preview. The store does not
  trust them to have redacted it: it redacts and truncates the preview itself,
  and refuses anything hash-shaped that is long enough to be text. A column
  that could hold a prompt does not exist, so no future caller can fill one.
* **One connection, one lock.** SQLite is opened with
  ``check_same_thread=False`` and every statement runs under a re-entrant lock,
  because the proxy touches this from threads borrowed by ``asyncio.to_thread``.
  Nothing here imports asyncio or blocks for long; the caller owns the thread.
* **Forward-only migrations.** ``PRAGMA user_version`` plus an append-only list
  of scripts, so an older database on disk upgrades in place at open instead of
  failing at the first unknown column.
* **Statistics in Python, not SQL.** Percentiles and the weighted means are
  computed here. SQLite has no percentile function, and a hand-rolled SQL one
  would be slower and less obvious than sorting a few thousand floats.

WAL and ``synchronous=NORMAL`` are chosen deliberately: a write must never
block the request that is being routed, and losing the last few rows to a power
cut costs nothing but a gap in a dashboard.
"""

from __future__ import annotations

import json
import math
import sqlite3
import threading
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from switchboard.privacy import preview as make_preview
from switchboard.registry import Measurement

RouteSource = Literal["proxy", "api", "mcp", "cli", "shadow"]
DecisionKind = Literal["task", "step"]
SwitchTrigger = Literal["auto", "explicit", "manual"]
MeasurementMetric = Literal["load_s", "gen_tps", "prompt_tps"]

ROUTE_SOURCES: tuple[RouteSource, ...] = ("proxy", "api", "mcp", "cli", "shadow")
DECISION_KINDS: tuple[DecisionKind, ...] = ("task", "step")
SWITCH_TRIGGERS: tuple[SwitchTrigger, ...] = ("auto", "explicit", "manual")
MEASUREMENT_METRICS: tuple[MeasurementMetric, ...] = ("load_s", "gen_tps", "prompt_tps")

# The newest sample carries this much of the weight; one ten samples back keeps
# 0.7**10 ~= 3%. Chosen so a model re-measured after a driver or context-size
# change converges within a handful of runs without a single outlier owning the
# number.
EWMA_ALPHA = 0.3

# How many of the most recent samples per (model, metric) the mean folds over.
# Beyond this the weights are noise, and the bound keeps the fold O(1) as the
# database grows.
EWMA_WINDOW = 200

# A hash is short. Anything longer is text that does not belong in this file.
MAX_HASH_CHARS = 64

# Hard ceiling on a stored preview, independent of what the caller asks for.
MAX_PREVIEW_CHARS = 200

SCHEMA_VERSION = 1

# PRIVACY CONTRACT FOR THIS SCHEMA:
# No column in this database may hold a raw prompt, response, or agent
# transcript. A request is identified by `prompt_hash` (switchboard.privacy
# .hash_text); the single human-readable excerpt is `routes.preview`, which the
# store redacts and truncates before writing. `reason`, `error` and
# `recommendation` hold strings Switchboard itself generates, never user text.
# Agent steps are keyed by `goal_hash`/`action_hash` for the same reason.
_SCHEMA_V1 = """
CREATE TABLE IF NOT EXISTS routes (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                      TEXT    NOT NULL,
    session_id              TEXT,
    source                  TEXT    NOT NULL
                            CHECK (source IN ('proxy','api','mcp','cli','shadow')),
    mode                    TEXT,
    policy                  TEXT,
    requested_model         TEXT,
    selected_model          TEXT,
    resident_before         TEXT,
    switched                INTEGER NOT NULL DEFAULT 0 CHECK (switched IN (0,1)),
    switch_ms               REAL,
    decision_backend        TEXT,
    decision_latency_ms     REAL,
    decision_json           TEXT,
    reason                  TEXT,
    confidence              REAL,
    prompt_hash             TEXT,
    preview                 TEXT,
    estimated_prompt_tokens INTEGER,
    streaming               INTEGER CHECK (streaming IN (0,1)),
    downstream_status       INTEGER,
    ttfb_ms                 REAL,
    downstream_ms           REAL,
    tokens_in               INTEGER,
    tokens_out              INTEGER,
    gen_tps                 REAL,
    error                   TEXT,
    outcome                 TEXT
);
CREATE INDEX IF NOT EXISTS idx_routes_ts ON routes (ts);
CREATE INDEX IF NOT EXISTS idx_routes_session ON routes (session_id, ts);

CREATE TABLE IF NOT EXISTS decision_attempts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            TEXT    NOT NULL,
    route_id      INTEGER REFERENCES routes (id) ON DELETE CASCADE,
    backend       TEXT    NOT NULL,
    kind          TEXT    NOT NULL CHECK (kind IN ('task','step')),
    ok            INTEGER NOT NULL CHECK (ok IN (0,1)),
    error_kind    TEXT,
    latency_ms    REAL,
    input_tokens  INTEGER,
    output_tokens INTEGER,
    est_cost_usd  REAL
);
CREATE INDEX IF NOT EXISTS idx_attempts_ts ON decision_attempts (ts);
CREATE INDEX IF NOT EXISTS idx_attempts_route ON decision_attempts (route_id);

CREATE TABLE IF NOT EXISTS switches (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts         TEXT    NOT NULL,
    from_model TEXT,
    to_model   TEXT,
    executor   TEXT,
    "trigger"  TEXT    CHECK ("trigger" IN ('auto','explicit','manual')),
    reason     TEXT,
    ok         INTEGER NOT NULL DEFAULT 0 CHECK (ok IN (0,1)),
    load_ms    REAL,
    error      TEXT
);
CREATE INDEX IF NOT EXISTS idx_switches_ts ON switches (ts);

CREATE TABLE IF NOT EXISTS step_evals (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                 TEXT    NOT NULL,
    session_id         TEXT,
    goal_hash          TEXT,
    action_hash        TEXT,
    attempt            INTEGER,
    outcome            TEXT,
    progress           REAL,
    complete           REAL,
    needs_new_approach REAL,
    recommendation     TEXT,
    backend            TEXT,
    latency_ms         REAL
);
CREATE INDEX IF NOT EXISTS idx_step_evals_session ON step_evals (session_id, ts);

CREATE TABLE IF NOT EXISTS measurements (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    ts       TEXT    NOT NULL,
    model_id TEXT    NOT NULL,
    metric   TEXT    NOT NULL CHECK (metric IN ('load_s','gen_tps','prompt_tps')),
    value    REAL    NOT NULL,
    source   TEXT,
    context  TEXT
);
CREATE INDEX IF NOT EXISTS idx_measurements_model ON measurements (model_id, metric, ts);

CREATE TABLE IF NOT EXISTS sessions (
    session_id      TEXT PRIMARY KEY,
    current_model   TEXT,
    last_switch_ts  TEXT,
    turns           INTEGER NOT NULL DEFAULT 0,
    difficulty_ewma REAL,
    created_ts      TEXT NOT NULL,
    updated_ts      TEXT NOT NULL
);
"""

# Append only. Each entry brings a database from the previous version to its own.
_MIGRATIONS: tuple[tuple[int, str], ...] = ((1, _SCHEMA_V1),)

# Columns `update_route` may write: everything that is only known once the
# downstream response is finished, plus the two switch fields, because a route
# may fall back to the resident model after the decision was recorded.
_ROUTE_UPDATABLE = frozenset(
    {
        "selected_model",
        "switched",
        "switch_ms",
        "downstream_status",
        "ttfb_ms",
        "downstream_ms",
        "tokens_in",
        "tokens_out",
        "gen_tps",
        "error",
        "outcome",
    }
)

_SESSION_UPDATABLE = frozenset(
    {"current_model", "last_switch_ts", "turns", "difficulty_ewma"}
)

# Written as 0/1 by SQLite, handed back as bool.
_BOOL_COLUMNS = frozenset({"switched", "streaming", "ok"})

# Stored as JSON text, handed back parsed under the key named here.
_JSON_COLUMNS: dict[str, str] = {"decision_json": "decision", "context": "context"}


def iso_utc(moment: datetime | None = None) -> str:
    """An ISO-8601 UTC timestamp in the one format this store compares.

    Fixed width and always ``Z``, so a lexicographic ``ts >= cutoff`` in SQL is
    a chronological comparison. Callers that supply their own ``ts`` must build
    it here, or the time-window queries will quietly mis-sort them.
    """
    stamp = (moment or datetime.now(UTC)).astimezone(UTC)
    return stamp.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def percentile(values: Sequence[float], fraction: float) -> float | None:
    """Nearest-rank percentile, or None for an empty sample.

    Nearest rank rather than interpolation: these are latencies, and a p95 that
    is a real observed request is easier to reason about than an average of two
    requests that never happened.
    """
    ordered = sorted(values)
    if not ordered:
        return None
    rank = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return float(ordered[rank])


def ewma(values: Sequence[float], alpha: float = EWMA_ALPHA) -> float | None:
    """Exponentially weighted mean of an oldest-first sequence.

    The weights sum to one, so the result stays in the range of the samples: it
    is a mean that trusts the newest measurement most, not a decaying sum.
    """
    if not values:
        return None
    result = float(values[0])
    for value in values[1:]:
        result = alpha * float(value) + (1.0 - alpha) * result
    return result


def _bool_column(value: bool | int | None) -> int | None:
    return None if value is None else int(bool(value))


def _json_column(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, default=str, separators=(",", ":"))


def _require_hash(name: str, value: str | None) -> str | None:
    """Reject anything passed as a hash that is long enough to be a prompt."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if len(text) > MAX_HASH_CHARS or "\n" in text:
        raise ValueError(
            f"{name} must be a hash (see switchboard.privacy.hash_text), not text; "
            f"got {len(text)} characters"
        )
    return text


def _require_choice(name: str, value: str, allowed: Iterable[str]) -> str:
    options = tuple(allowed)
    if value not in options:
        raise ValueError(f"{name} must be one of {options}, got {value!r}")
    return value


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    """A row as plain data: 0/1 back to bool, JSON text back to objects."""
    data: dict[str, Any] = {}
    for key in row.keys():
        value = row[key]
        if key in _BOOL_COLUMNS:
            data[key] = None if value is None else bool(value)
        elif key in _JSON_COLUMNS:
            parsed: Any = None
            if value:
                try:
                    parsed = json.loads(value)
                except json.JSONDecodeError:
                    # Written by an older version, or by hand. Keep the text
                    # rather than losing the row.
                    parsed = value
            data[_JSON_COLUMNS[key]] = parsed
        else:
            data[key] = value
    return data


class Store:
    """SQLite-backed observability for one Switchboard process.

    Safe to call from a thread pool: every statement runs under a lock, so an
    async caller wraps these methods in ``asyncio.to_thread`` and nothing else.
    Construct with ``Path(":memory:")`` for a throwaway database in tests.
    """

    def __init__(self, path: Path, *, preview_chars: int = 80) -> None:
        self.path = path
        # A caller's preview_chars only ever tightens the store's own ceiling.
        self.preview_chars = max(1, min(int(preview_chars), MAX_PREVIEW_CHARS))
        self._lock = threading.RLock()
        self._closed = False

        target = str(path)
        if target != ":memory:":
            path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(target, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        # WAL keeps a dashboard read from blocking the write on the hot path.
        # It is a no-op for an in-memory database, which reports 'memory'.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._migrate()

    # --- lifecycle ---------------------------------------------------------

    def _migrate(self) -> None:
        with self._lock:
            version = int(self._conn.execute("PRAGMA user_version").fetchone()[0])
            for target, script in _MIGRATIONS:
                if target <= version:
                    continue
                self._conn.executescript(script)
                # PRAGMA cannot be parameterised; the value is a literal from
                # _MIGRATIONS, never from a caller.
                self._conn.execute(f"PRAGMA user_version = {int(target)}")
                version = target
            self._conn.commit()

    @property
    def schema_version(self) -> int:
        with self._lock:
            return int(self._conn.execute("PRAGMA user_version").fetchone()[0])

    def close(self) -> None:
        """Close the connection. Idempotent, so shutdown paths can be sloppy."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # --- low-level helpers -------------------------------------------------

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Cursor]:
        """A cursor under the lock, committed on success."""
        with self._lock:
            cursor = self._conn.cursor()
            try:
                yield cursor
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def _query(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, tuple(params)).fetchall()

    def _insert(self, table: str, values: Mapping[str, Any]) -> int:
        # Table and column names come from this module's constants and from
        # validated whitelists, never from caller-supplied strings.
        columns = ", ".join(f'"{name}"' for name in values)
        placeholders = ", ".join("?" for _ in values)
        sql = f'INSERT INTO "{table}" ({columns}) VALUES ({placeholders})'
        with self._write() as cursor:
            cursor.execute(sql, tuple(values.values()))
            return int(cursor.lastrowid or 0)

    # --- routes ------------------------------------------------------------

    def record_route(
        self,
        *,
        source: RouteSource,
        session_id: str | None = None,
        mode: str | None = None,
        policy: str | None = None,
        requested_model: str | None = None,
        selected_model: str | None = None,
        resident_before: str | None = None,
        switched: bool = False,
        switch_ms: float | None = None,
        decision_backend: str | None = None,
        decision_latency_ms: float | None = None,
        decision: Mapping[str, Any] | None = None,
        reason: str | None = None,
        confidence: float | None = None,
        prompt_hash: str | None = None,
        preview: str | None = None,
        estimated_prompt_tokens: int | None = None,
        streaming: bool | None = None,
        ts: str | None = None,
    ) -> int:
        """Record a routing decision, returning the route id to update later.

        ``decision`` is a decision ``summary()`` dict and is stored as JSON.
        ``preview`` is redacted and truncated here regardless of what the caller
        did; ``prompt_hash`` is rejected if it is long enough to be text.
        """
        values = {
            "ts": ts or iso_utc(),
            "session_id": session_id,
            "source": _require_choice("source", source, ROUTE_SOURCES),
            "mode": mode,
            "policy": policy,
            "requested_model": requested_model,
            "selected_model": selected_model,
            "resident_before": resident_before,
            "switched": _bool_column(switched),
            "switch_ms": switch_ms,
            "decision_backend": decision_backend,
            "decision_latency_ms": decision_latency_ms,
            "decision_json": _json_column(dict(decision) if decision is not None else None),
            "reason": reason,
            "confidence": confidence,
            "prompt_hash": _require_hash("prompt_hash", prompt_hash),
            "preview": self._safe_preview(preview),
            "estimated_prompt_tokens": estimated_prompt_tokens,
            "streaming": _bool_column(streaming),
        }
        return self._insert("routes", values)

    def _safe_preview(self, text: str | None) -> str | None:
        """Redact and shorten, so the column cannot hold a prompt even by mistake."""
        if text is None:
            return None
        cleaned = make_preview(text, limit=self.preview_chars)
        return cleaned or None

    def update_route(self, route_id: int, **fields: Any) -> bool:
        """Fill in what was only knowable once the response finished.

        Unknown column names raise rather than being dropped: a typo here would
        otherwise lose a metric silently for as long as nobody looked.
        """
        if not fields:
            return False
        unknown = set(fields) - _ROUTE_UPDATABLE
        if unknown:
            raise ValueError(
                f"update_route cannot set {sorted(unknown)}; "
                f"allowed: {sorted(_ROUTE_UPDATABLE)}"
            )
        prepared = {
            key: _bool_column(value) if key in _BOOL_COLUMNS else value
            for key, value in fields.items()
        }
        assignments = ", ".join(f'"{name}" = ?' for name in prepared)
        with self._write() as cursor:
            cursor.execute(
                f"UPDATE routes SET {assignments} WHERE id = ?",
                (*prepared.values(), int(route_id)),
            )
            return cursor.rowcount > 0

    def get_route(self, route_id: int) -> dict[str, Any] | None:
        rows = self._query("SELECT * FROM routes WHERE id = ?", (int(route_id),))
        return _row_to_dict(rows[0]) if rows else None

    def recent_routes(self, limit: int = 50) -> list[dict[str, Any]]:
        """The newest routes first, decision JSON already parsed."""
        rows = self._query(
            "SELECT * FROM routes ORDER BY id DESC LIMIT ?", (max(0, int(limit)),)
        )
        return [_row_to_dict(row) for row in rows]

    # --- decision attempts -------------------------------------------------

    def record_attempt(
        self,
        *,
        backend: str,
        kind: DecisionKind,
        ok: bool,
        route_id: int | None = None,
        error_kind: str | None = None,
        latency_ms: float | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        est_cost_usd: float | None = None,
        ts: str | None = None,
    ) -> int:
        """Record one backend attempt, successful or not.

        ``route_id`` is nullable because step evaluations and shadow decisions
        have no route, and because the attempt is written before the route row
        exists in some paths.
        """
        values = {
            "ts": ts or iso_utc(),
            "route_id": int(route_id) if route_id is not None else None,
            "backend": backend,
            "kind": _require_choice("kind", kind, DECISION_KINDS),
            "ok": _bool_column(ok),
            "error_kind": error_kind,
            "latency_ms": latency_ms,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "est_cost_usd": est_cost_usd,
        }
        return self._insert("decision_attempts", values)

    # --- switches ----------------------------------------------------------

    def record_switch(
        self,
        *,
        from_model: str | None,
        to_model: str | None,
        executor: str | None = None,
        trigger: SwitchTrigger = "auto",
        reason: str | None = None,
        ok: bool = False,
        load_ms: float | None = None,
        error: str | None = None,
        ts: str | None = None,
    ) -> int:
        """Record a model switch attempt. Failed switches matter most, so ok=False
        is the default: a caller that never confirms leaves a visible failure."""
        values = {
            "ts": ts or iso_utc(),
            "from_model": from_model,
            "to_model": to_model,
            "executor": executor,
            "trigger": _require_choice("trigger", trigger, SWITCH_TRIGGERS),
            "reason": reason,
            "ok": _bool_column(ok),
            "load_ms": load_ms,
            "error": error,
        }
        return self._insert("switches", values)

    def recent_switches(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self._query(
            "SELECT * FROM switches ORDER BY id DESC LIMIT ?", (max(0, int(limit)),)
        )
        return [_row_to_dict(row) for row in rows]

    # --- step evaluations --------------------------------------------------

    def record_step_eval(
        self,
        *,
        session_id: str | None = None,
        goal_hash: str | None = None,
        action_hash: str | None = None,
        attempt: int = 1,
        outcome: str | None = None,
        progress: float | None = None,
        complete: float | None = None,
        needs_new_approach: float | None = None,
        recommendation: str | None = None,
        backend: str | None = None,
        latency_ms: float | None = None,
        ts: str | None = None,
    ) -> int:
        """Record one supervised agent step. Goal and action arrive hashed."""
        values = {
            "ts": ts or iso_utc(),
            "session_id": session_id,
            "goal_hash": _require_hash("goal_hash", goal_hash),
            "action_hash": _require_hash("action_hash", action_hash),
            "attempt": int(attempt),
            "outcome": outcome,
            "progress": progress,
            "complete": complete,
            "needs_new_approach": needs_new_approach,
            "recommendation": recommendation,
            "backend": backend,
            "latency_ms": latency_ms,
        }
        return self._insert("step_evals", values)

    def recent_step_evals(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self._query(
            "SELECT * FROM step_evals ORDER BY id DESC LIMIT ?", (max(0, int(limit)),)
        )
        return [_row_to_dict(row) for row in rows]

    # --- measurements ------------------------------------------------------

    def record_measurement(
        self,
        *,
        model_id: str,
        metric: MeasurementMetric,
        value: float,
        source: str | None = None,
        context: Mapping[str, Any] | None = None,
        ts: str | None = None,
    ) -> int:
        """Record one observed number for a model, with optional JSON context."""
        values = {
            "ts": ts or iso_utc(),
            "model_id": model_id,
            "metric": _require_choice("metric", metric, MEASUREMENT_METRICS),
            "value": float(value),
            "source": source,
            "context": _json_column(dict(context) if context is not None else None),
        }
        return self._insert("measurements", values)

    def measurements_by_model(self) -> dict[str, dict[str, Measurement]]:
        """Runtime numbers for the registry, keyed by model id then metric.

        Samples are folded newest-last with :data:`EWMA_ALPHA`, over at most the
        most recent :data:`EWMA_WINDOW` of them. The resulting Measurement is
        flagged measured, because everything in this table was observed on this
        machine, and carries how many samples stand behind it.
        """
        rows = self._query(
            "SELECT model_id, metric, value, ts FROM measurements "
            "ORDER BY model_id, metric, ts ASC, id ASC"
        )
        grouped: dict[tuple[str, str], list[tuple[float, str]]] = {}
        for row in rows:
            grouped.setdefault((row["model_id"], row["metric"]), []).append(
                (float(row["value"]), str(row["ts"]))
            )

        result: dict[str, dict[str, Measurement]] = {}
        for (model_id, metric), samples in grouped.items():
            window = samples[-EWMA_WINDOW:]
            value = ewma([sample for sample, _ in window])
            if value is None:
                continue
            latest = window[-1][1][:10]
            result.setdefault(model_id, {})[metric] = Measurement(
                value=value,
                source=f"measured on this machine (n={len(window)}, latest {latest})",
                measured=True,
                samples=len(window),
            )
        return result

    # --- sessions ----------------------------------------------------------

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        rows = self._query(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        )
        return _row_to_dict(rows[0]) if rows else None

    def upsert_session(
        self, session_id: str, *, bump_turns: bool = False, **fields: Any
    ) -> dict[str, Any]:
        """Create or merge a session row, returning it as it now stands.

        Merge, not replace: a field the caller omits keeps its stored value, so
        the turn counter survives a call that only moves the current model.
        ``bump_turns`` increments in SQL rather than read-modify-write, because
        two requests in one conversation can land on two threads.
        """
        unknown = set(fields) - _SESSION_UPDATABLE
        if unknown:
            raise ValueError(
                f"upsert_session cannot set {sorted(unknown)}; "
                f"allowed: {sorted(_SESSION_UPDATABLE)}"
            )
        now = iso_utc()
        with self._write() as cursor:
            cursor.execute(
                "INSERT INTO sessions (session_id, turns, created_ts, updated_ts) "
                "VALUES (?, 0, ?, ?) ON CONFLICT(session_id) DO NOTHING",
                (session_id, now, now),
            )
            assignments = [f'"{name}" = ?' for name in fields]
            params: list[Any] = list(fields.values())
            if bump_turns:
                assignments.append('"turns" = "turns" + 1')
            assignments.append('"updated_ts" = ?')
            params.append(now)
            params.append(session_id)
            cursor.execute(
                f"UPDATE sessions SET {', '.join(assignments)} WHERE session_id = ?",
                tuple(params),
            )
        session = self.get_session(session_id)
        # The row was just written under the same lock; absence would be a bug.
        assert session is not None
        return session

    # --- statistics --------------------------------------------------------

    def _cutoff(self, hours: float) -> str | None:
        """The ISO timestamp a window starts at, or None for 'everything'."""
        if hours <= 0:
            return None
        return iso_utc(datetime.now(UTC) - timedelta(hours=hours))

    def provider_stats(self, hours: float = 24) -> dict[str, dict[str, Any]]:
        """Per-backend health and cost over the window.

        This is what decides whether a cloud backend earns its place: attempts,
        how many answered, what the failures were, the latency spread, and what
        it cost.
        """
        cutoff = self._cutoff(hours)
        sql = (
            "SELECT backend, ok, error_kind, latency_ms, est_cost_usd "
            "FROM decision_attempts"
        )
        params: tuple[Any, ...] = ()
        if cutoff is not None:
            sql += " WHERE ts >= ?"
            params = (cutoff,)

        stats: dict[str, dict[str, Any]] = {}
        latencies: dict[str, list[float]] = {}
        for row in self._query(sql, params):
            backend = row["backend"]
            entry = stats.setdefault(
                backend,
                {
                    "attempts": 0,
                    "ok": 0,
                    "error_kinds": {},
                    "p50_latency_ms": None,
                    "p95_latency_ms": None,
                    "est_cost_usd": 0.0,
                },
            )
            entry["attempts"] += 1
            if row["ok"]:
                entry["ok"] += 1
            else:
                kind = row["error_kind"] or "unknown"
                entry["error_kinds"][kind] = entry["error_kinds"].get(kind, 0) + 1
            if row["latency_ms"] is not None:
                latencies.setdefault(backend, []).append(float(row["latency_ms"]))
            if row["est_cost_usd"] is not None:
                entry["est_cost_usd"] += float(row["est_cost_usd"])

        for backend, entry in stats.items():
            samples = latencies.get(backend, [])
            entry["p50_latency_ms"] = percentile(samples, 0.50)
            entry["p95_latency_ms"] = percentile(samples, 0.95)
            entry["est_cost_usd"] = round(entry["est_cost_usd"], 6)
        return stats

    def route_stats(self, hours: float = 24) -> dict[str, Any]:
        """Routing shape over the window: what was chosen, by whom, how fast.

        ``fallbacks`` counts routes where at least one decision backend failed
        and the chain had to fall through to another one, which is the number
        that says whether the preferred backend is actually carrying traffic.
        """
        cutoff = self._cutoff(hours)
        where = " WHERE ts >= ?" if cutoff is not None else ""
        params: tuple[Any, ...] = (cutoff,) if cutoff is not None else ()

        rows = self._query(
            "SELECT id, selected_model, decision_backend, switched, streaming, "
            f"decision_latency_ms, downstream_ms FROM routes{where}",
            params,
        )

        by_model: dict[str, int] = {}
        by_backend: dict[str, int] = {}
        decision_latencies: list[float] = []
        downstream_latencies: list[float] = []
        switches = 0
        streaming = 0
        for row in rows:
            model = row["selected_model"] or "unrouted"
            by_model[model] = by_model.get(model, 0) + 1
            backend = row["decision_backend"] or "none"
            by_backend[backend] = by_backend.get(backend, 0) + 1
            if row["switched"]:
                switches += 1
            if row["streaming"]:
                streaming += 1
            if row["decision_latency_ms"] is not None:
                decision_latencies.append(float(row["decision_latency_ms"]))
            if row["downstream_ms"] is not None:
                downstream_latencies.append(float(row["downstream_ms"]))

        fallback_rows = self._query(
            "SELECT COUNT(DISTINCT a.route_id) FROM decision_attempts a "
            "JOIN routes r ON r.id = a.route_id "
            f"WHERE a.ok = 0{' AND r.ts >= ?' if cutoff is not None else ''}",
            params,
        )
        fallbacks = int(fallback_rows[0][0]) if fallback_rows else 0

        total = len(rows)
        return {
            "routes": total,
            "by_model": by_model,
            "by_backend": by_backend,
            "switches": switches,
            "fallbacks": fallbacks,
            "p50_decision_ms": percentile(decision_latencies, 0.50),
            "p95_decision_ms": percentile(decision_latencies, 0.95),
            "p50_downstream_ms": percentile(downstream_latencies, 0.50),
            "p95_downstream_ms": percentile(downstream_latencies, 0.95),
            "streaming_share": (streaming / total) if total else 0.0,
        }

    # --- maintenance -------------------------------------------------------

    def prune(self, retention_days: int) -> dict[str, int]:
        """Drop rows older than the retention window; return what went.

        ``measurements`` are deliberately never pruned: they are the only record
        of how this machine actually performs, the registry reads them on every
        build, and a few thousand floats cost nothing. Sessions go once they
        have been idle for the window, since a conversation that old will not
        come back.
        """
        if retention_days <= 0:
            raise ValueError("retention_days must be positive")
        cutoff = iso_utc(datetime.now(UTC) - timedelta(days=retention_days))
        deleted: dict[str, int] = {}
        with self._write() as cursor:
            # Attempts first, and including those that hang off a route about to
            # go, so the reported counts are real rather than hidden in a cascade.
            cursor.execute(
                "DELETE FROM decision_attempts WHERE ts < ? OR route_id IN "
                "(SELECT id FROM routes WHERE ts < ?)",
                (cutoff, cutoff),
            )
            deleted["decision_attempts"] = cursor.rowcount
            for table, column in (
                ("routes", "ts"),
                ("switches", "ts"),
                ("step_evals", "ts"),
                ("sessions", "updated_ts"),
            ):
                cursor.execute(f'DELETE FROM "{table}" WHERE "{column}" < ?', (cutoff,))
                deleted[table] = cursor.rowcount
        return deleted

    def vacuum(self) -> None:
        """Reclaim file space after a large prune. Blocks writers; call off-peak."""
        with self._lock:
            self._conn.execute("VACUUM")
