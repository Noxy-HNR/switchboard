"""The backend contract.

A decision backend answers the questions in `schema` and nothing else. It knows
how to reach its provider, how to fail, and what state it may see; it knows
nothing about models, routing or SQLite.

Three rules every backend follows:

* **Say when you cannot help.** `availability()` reports `unconfigured` (no
  credential), `unavailable` (endpoint down, too slow, breaker open) or `ready`,
  so the chain can skip it without paying a timeout.
* **Raise, do not guess.** On any failure raise `BackendError`; never return a
  half-filled decision that the policy would mistake for a real judgment.
* **Own your privacy boundary.** A backend that sends data off the machine
  redacts and truncates it itself, from the raw context it is handed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

from switchboard.decisions.schema import StepDecision, TaskDecision


class BackendError(RuntimeError):
    """A backend could not produce a decision.

    `kind` classifies the failure for the circuit breaker and for the provider
    health view: a timeout or outage should open the breaker, while a refusal to
    run (no credential, resident model too slow) should not count against it.
    """

    def __init__(
        self,
        message: str,
        *,
        backend: str,
        kind: Literal[
            "timeout", "connection", "auth", "rate_limit", "server", "invalid", "skipped"
        ] = "server",
        retryable: bool = True,
    ) -> None:
        super().__init__(message)
        self.backend = backend
        self.kind = kind
        self.retryable = retryable

    @property
    def counts_against_health(self) -> bool:
        return self.kind not in {"skipped", "auth"}


AvailabilityState = Literal["ready", "unconfigured", "unavailable"]


@dataclass(slots=True)
class Availability:
    state: AvailabilityState
    reason: str = ""

    @property
    def ready(self) -> bool:
        return self.state == "ready"

    @classmethod
    def ok(cls) -> "Availability":
        return cls(state="ready")

    @classmethod
    def unconfigured(cls, reason: str) -> "Availability":
        return cls(state="unconfigured", reason=reason)

    @classmethod
    def unavailable(cls, reason: str) -> "Availability":
        return cls(state="unavailable", reason=reason)


@dataclass(slots=True)
class BackendFacts:
    """What a backend needs to know about the local stack to decide if it can run.

    Passed in rather than looked up so backends stay testable and so the chain
    reads the upstream state once per request instead of once per backend.
    """

    resident_model_id: str | None = None
    resident_gen_tps: float | None = None
    server_healthy: bool = False
    server_busy: bool = False


@dataclass(slots=True)
class TaskContext:
    """The raw material for a task judgment.

    Holds unredacted text: each backend applies the privacy rules its own
    destination requires. Nothing here is stored; the store keeps hashes.
    """

    message: str
    prior_message: str | None = None
    system_prompt: str | None = None
    tool_names: list[str] = field(default_factory=list)
    turn_index: int = 0
    estimated_prompt_tokens: int = 0
    has_images: bool = False
    # Stable per conversation, already hashed by the caller.
    session_id: str | None = None

    def cache_key(self) -> tuple[str, ...]:
        return (
            self.message.strip(),
            self.prior_message or "",
            ",".join(sorted(self.tool_names)),
            str(self.has_images),
        )


@dataclass(slots=True)
class StepContext:
    """The raw material for a step judgment."""

    goal: str
    action: str
    result: str
    # Facts the caller knows exactly; backends may see them but must not invent them.
    exit_code: int | None = None
    attempt: int = 1
    previous_attempts: list[str] = field(default_factory=list)
    elapsed_s: float | None = None
    session_id: str | None = None

    def cache_key(self) -> tuple[str, ...]:
        # Everything a backend is shown is part of the key: the same step text
        # with a different exit code or retry history is a different question.
        return (
            self.goal.strip(),
            self.action.strip(),
            self.result.strip(),
            str(self.attempt),
            str(self.exit_code),
            "␟".join(item.strip() for item in self.previous_attempts),
            "" if self.elapsed_s is None else f"{self.elapsed_s:.3f}",
        )


@runtime_checkable
class DecisionBackend(Protocol):
    """One semantic decision provider."""

    name: str

    async def availability(self, facts: BackendFacts) -> Availability: ...

    async def decide_task(self, context: TaskContext, facts: BackendFacts) -> TaskDecision: ...

    async def decide_step(self, context: StepContext, facts: BackendFacts) -> StepDecision: ...

    async def aclose(self) -> None: ...


class Stopwatch:
    """Milliseconds elapsed, for the latency fields every decision carries."""

    __slots__ = ("_start",)

    def __init__(self) -> None:
        self._start = time.perf_counter()

    @property
    def ms(self) -> float:
        return (time.perf_counter() - self._start) * 1000.0
