"""Ordered fallback across decision backends.

The rule is simple: try the configured backends in order, take the first real
answer, and never let a decision provider make the local stack unusable. That
means:

* **A budget, not just timeouts.** The whole semantic layer gets
  ``decisions.budget_ms``. Each backend may use its own timeout or what is left
  of the budget, whichever is smaller, so a slow cloud provider cannot push a
  chat request past the point where the answer matters.
* **A breaker.** After a few failures a backend is skipped for a cooldown
  instead of being retried on every request. Refusals ("no API key", "resident
  model too slow") are not failures and never open the breaker.
* **A cache.** The same question about the same text within the TTL reuses the
  answer, which matters because clients retry and agents re-ask.
* **A floor.** The last backend in the chain is the rule engine, which cannot
  fail, so there is always a decision.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

from switchboard.config import Config
from switchboard.decisions.base import (
    Availability,
    BackendError,
    BackendFacts,
    DecisionBackend,
    StepContext,
    TaskContext,
)
from switchboard.decisions.schema import StepDecision, TaskDecision

Kind = Literal["task", "step"]


@dataclass(slots=True)
class Attempt:
    """One call to one backend, successful or not."""

    backend: str
    kind: Kind
    ok: bool
    latency_ms: float
    error_kind: str | None = None
    error: str | None = None
    skipped_reason: str | None = None
    usage: dict[str, int] = field(default_factory=dict)

    def describe(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "ok": self.ok,
            "latency_ms": round(self.latency_ms, 1),
            "error_kind": self.error_kind,
            "error": self.error,
            "skipped_reason": self.skipped_reason,
            "usage": self.usage or None,
        }


@dataclass(slots=True)
class DecisionResult:
    """A decision plus the story of how it was reached."""

    decision: Any
    attempts: list[Attempt] = field(default_factory=list)
    cached: bool = False

    @property
    def backend(self) -> str:
        return getattr(self.decision, "backend", "unknown")

    @property
    def fell_back(self) -> bool:
        """Whether anything ahead of the winning backend failed or refused."""
        return any(not attempt.ok for attempt in self.attempts)

    def describe(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "cached": self.cached,
            "fell_back": self.fell_back,
            "attempts": [attempt.describe() for attempt in self.attempts],
        }


class Breaker:
    """Per-backend failure counter with a cooldown."""

    __slots__ = ("failures", "opened_until", "last_error", "threshold", "cooldown_s")

    def __init__(self, threshold: int, cooldown_s: float) -> None:
        self.threshold = threshold
        self.cooldown_s = cooldown_s
        self.failures = 0
        self.opened_until = 0.0
        self.last_error: str | None = None

    def open(self, now: float) -> bool:
        return now < self.opened_until

    def record_success(self) -> None:
        self.failures = 0
        self.opened_until = 0.0

    def record_failure(self, error: BackendError, now: float) -> None:
        if not error.counts_against_health:
            return
        self.failures += 1
        self.last_error = f"{error.kind}: {error}"
        if self.failures >= self.threshold:
            self.opened_until = now + self.cooldown_s

    def describe(self, now: float) -> dict[str, Any]:
        return {
            "consecutive_failures": self.failures,
            "open": self.open(now),
            "reopens_in_s": max(0.0, round(self.opened_until - now, 1)) if self.open(now) else 0.0,
            "last_error": self.last_error,
        }


class _Cache:
    """Tiny TTL cache keyed by the question and the text it is about."""

    def __init__(self, ttl_s: float, max_entries: int) -> None:
        self._ttl_s = ttl_s
        self._max_entries = max_entries
        self._entries: dict[tuple[Any, ...], tuple[float, Any]] = {}

    def get(self, key: tuple[Any, ...]) -> Any | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        stored_at, value = entry
        if time.monotonic() - stored_at > self._ttl_s:
            self._entries.pop(key, None)
            return None
        return value

    def put(self, key: tuple[Any, ...], value: Any) -> None:
        if len(self._entries) >= self._max_entries:
            # Cheap eviction: drop the oldest quarter rather than tracking LRU.
            for stale in sorted(self._entries, key=lambda k: self._entries[k][0])[
                : max(1, self._max_entries // 4)
            ]:
                self._entries.pop(stale, None)
        self._entries[key] = (time.monotonic(), value)

    def clear(self) -> None:
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)


class DecisionChain:
    """Runs the configured backends in order until one answers."""

    def __init__(self, config: Config, backends: Sequence[DecisionBackend]) -> None:
        self._config = config
        self._backends = list(backends)
        jev = config.decisions.jev
        self._breakers = {
            backend.name: Breaker(jev.breaker_failures, jev.breaker_cooldown_s)
            for backend in self._backends
        }
        self._cache = _Cache(config.decisions.cache_ttl_s, config.decisions.cache_max_entries)
        self._availability: dict[str, Availability] = {}

    @property
    def backends(self) -> list[DecisionBackend]:
        return list(self._backends)

    async def aclose(self) -> None:
        for backend in self._backends:
            try:
                await backend.aclose()
            except Exception:  # noqa: BLE001 - shutdown must not raise
                pass

    async def decide_task(
        self, context: TaskContext, facts: BackendFacts, *, use_cache: bool = True
    ) -> DecisionResult:
        return await self._run("task", context, facts, use_cache=use_cache)

    async def decide_step(
        self, context: StepContext, facts: BackendFacts, *, use_cache: bool = True
    ) -> DecisionResult:
        return await self._run("step", context, facts, use_cache=use_cache)

    async def _run(
        self,
        kind: Kind,
        context: TaskContext | StepContext,
        facts: BackendFacts,
        *,
        use_cache: bool,
    ) -> DecisionResult:
        key = (kind, *context.cache_key())
        if use_cache:
            cached = self._cache.get(key)
            if cached is not None:
                return DecisionResult(decision=cached, cached=True)

        attempts: list[Attempt] = []
        deadline = time.monotonic() + self._config.decisions.budget_ms / 1000.0
        last_error: BackendError | None = None

        for backend in self._backends:
            now = time.monotonic()
            breaker = self._breakers[backend.name]

            if breaker.open(now):
                attempts.append(
                    Attempt(
                        backend=backend.name,
                        kind=kind,
                        ok=False,
                        latency_ms=0.0,
                        error_kind="breaker",
                        skipped_reason=f"breaker open after {breaker.failures} failures",
                    )
                )
                continue

            remaining = deadline - now
            # The rule engine is local, instant and cannot fail, so it is allowed
            # to run even when the budget is spent: an expired budget must still
            # produce a decision.
            is_floor = backend is self._backends[-1]
            if remaining <= 0 and not is_floor:
                attempts.append(
                    Attempt(
                        backend=backend.name,
                        kind=kind,
                        ok=False,
                        latency_ms=0.0,
                        error_kind="budget",
                        skipped_reason="decision budget exhausted",
                    )
                )
                continue

            availability = await backend.availability(facts)
            self._availability[backend.name] = availability
            if not availability.ready:
                attempts.append(
                    Attempt(
                        backend=backend.name,
                        kind=kind,
                        ok=False,
                        latency_ms=0.0,
                        error_kind=availability.state,
                        skipped_reason=availability.reason,
                    )
                )
                continue

            started = time.perf_counter()
            try:
                decision = await self._call(backend, kind, context, facts, remaining, is_floor)
            except BackendError as error:
                latency = (time.perf_counter() - started) * 1000
                breaker.record_failure(error, time.monotonic())
                last_error = error
                attempts.append(
                    Attempt(
                        backend=backend.name,
                        kind=kind,
                        ok=False,
                        latency_ms=latency,
                        error_kind=error.kind,
                        error=str(error)[:300],
                    )
                )
                continue
            except Exception as error:  # noqa: BLE001 - a backend bug must not break routing
                latency = (time.perf_counter() - started) * 1000
                attempts.append(
                    Attempt(
                        backend=backend.name,
                        kind=kind,
                        ok=False,
                        latency_ms=latency,
                        error_kind="unexpected",
                        error=f"{type(error).__name__}: {error}"[:300],
                    )
                )
                continue

            breaker.record_success()
            attempts.append(
                Attempt(
                    backend=backend.name,
                    kind=kind,
                    ok=True,
                    latency_ms=decision.latency_ms,
                    usage=dict(decision.usage),
                )
            )
            if use_cache:
                self._cache.put(key, decision)
            return DecisionResult(decision=decision, attempts=attempts)

        # Every backend refused or failed, including the floor. Callers get a
        # default decision rather than an exception: a chat request must still
        # be answerable.
        fallback: Any = TaskDecision() if kind == "task" else StepDecision()
        fallback.backend = "none"
        fallback.degraded = True
        fallback.notes.append(
            f"no decision backend answered ({last_error or 'all backends unavailable'})"
        )
        return DecisionResult(decision=fallback, attempts=attempts)

    async def _call(
        self,
        backend: DecisionBackend,
        kind: Kind,
        context: TaskContext | StepContext,
        facts: BackendFacts,
        remaining_s: float,
        is_floor: bool,
    ) -> Any:
        """Invoke one backend under the smaller of its timeout and the budget."""
        if kind == "task":
            coroutine = backend.decide_task(context, facts)  # type: ignore[arg-type]
        else:
            coroutine = backend.decide_step(context, facts)  # type: ignore[arg-type]

        if is_floor:
            return await coroutine
        try:
            return await asyncio.wait_for(coroutine, timeout=max(0.05, remaining_s))
        except TimeoutError as error:
            raise BackendError(
                f"exceeded the remaining decision budget ({remaining_s * 1000:.0f} ms)",
                backend=backend.name,
                kind="timeout",
            ) from error

    def health(self) -> dict[str, dict[str, Any]]:
        """Per-backend view for `switchboard providers` and /status."""
        now = time.monotonic()
        report: dict[str, dict[str, Any]] = {}
        for backend in self._backends:
            availability = self._availability.get(backend.name)
            report[backend.name] = {
                "availability": availability.state if availability else "unknown",
                "reason": availability.reason if availability else "not yet probed",
                **self._breakers[backend.name].describe(now),
            }
        return report

    async def probe(self, facts: BackendFacts) -> dict[str, dict[str, Any]]:
        """Refresh availability for every backend without making a decision."""
        for backend in self._backends:
            try:
                self._availability[backend.name] = await backend.availability(facts)
            except Exception as error:  # noqa: BLE001
                self._availability[backend.name] = Availability.unavailable(
                    f"{type(error).__name__}: {error}"
                )
        return self.health()

    def clear_cache(self) -> None:
        self._cache.clear()
