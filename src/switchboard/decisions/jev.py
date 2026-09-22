"""The TypeSafe (Jev) cloud decision backend.

Why this exists: Jev answers every question in `schema` in one round trip, typed
and calibrated, for a few millionths of a dollar and in far less time than the
resident 27B needs to produce the same judgment - especially while that model is
busy answering the user. It is the first backend the chain tries and the only
one that leaves this machine, so it owns its privacy boundary outright.

The rules it follows:

* **One call per judgment.** All four questions for a task (or a step) travel in
  a single ``POST /v1/systemone``. The state is read once and the questions are
  answered in parallel, so four questions cost barely more than one - cheaper
  and faster than asking them separately.
* **Named state, nothing else.** The state is a small JSON object whose field
  paths are exactly the ones the shared question text names. Unrelated context
  costs accuracy, so the system prompt, the session id and the token estimate
  are deliberately left at home.
* **Minimize, then redact, before send.** Pasted machine output - tracebacks,
  code blocks, terminal runs, key material, whole JSON documents - is replaced
  by a one-line description and never leaves the machine; the user's own words
  around it still go, because that is what the judgments are about. What is
  left is redacted and cut to one shared character budget. *Which kinds* of
  content were withheld is recorded in the decision notes; the content itself
  is recorded nowhere. Error
  messages are built from provider metadata (status, request id) rather than
  response bodies, so nothing echoed back can smuggle prompt text into a log.
* **Raise, never guess.** A missing, mistyped or out-of-range answer is a
  failure, not a default. A half-filled decision that the policy mistook for a
  real judgment would be worse than no decision at all, so `degraded` is never
  set here: either the whole judgment parsed or `BackendError` was raised.

The circuit breaker (``decisions.jev.breaker_*``) is deliberately absent.
`availability()` is a statement about configuration only and never touches the
network, so the chain can ask every backend on every request; counting failures
and holding a cooldown belongs to whoever owns the chain's health view.
"""

from __future__ import annotations

from typing import Any, Final, Literal, cast

import httpx2
from typesafe_sdk import (
    AsyncTypeSafeClient,
    Choice,
    ChoiceAnswer,
    Noul,
    Question,
    RetryPolicy,
    Score,
    ScoreAnswer,
    SystemOneResponse,
    TypeSafeAPIConnectionError,
    TypeSafeAPIError,
    TypeSafeAPIResponseValidationError,
    TypeSafeAPITimeoutError,
    TypeSafeAuthenticationError,
    TypeSafeError,
    TypeSafeInternalServerError,
    TypeSafeRateLimitError,
    TypeSafeUnprocessableEntityError,
    Usage,
)

from switchboard.config import Config, PrivacyConfig
from switchboard.decisions.base import (
    Availability,
    BackendError,
    BackendFacts,
    StepContext,
    Stopwatch,
    TaskContext,
)
from switchboard.decisions.schema import (
    AMBIGUITY_CRITERIA,
    AMBIGUITY_INSTRUCTIONS,
    DIFFICULTY_CRITERIA,
    DIFFICULTY_INSTRUCTIONS,
    DOMAIN_CRITERIA,
    DOMAIN_INSTRUCTIONS,
    DOMAINS,
    STEP_COMPLETE_CRITERIA,
    STEP_COMPLETE_INSTRUCTIONS,
    STEP_NEW_APPROACH_CRITERIA,
    STEP_NEW_APPROACH_INSTRUCTIONS,
    STEP_OUTCOME_CRITERIA,
    STEP_OUTCOME_INSTRUCTIONS,
    STEP_OUTCOMES,
    STEP_PROGRESS_CRITERIA,
    STEP_PROGRESS_INSTRUCTIONS,
    TOOL_NEED_CRITERIA,
    TOOL_NEED_INSTRUCTIONS,
    Domain,
    StepDecision,
    StepOutcome,
    TaskDecision,
)
from switchboard.privacy import minimize_for_cloud, prepare_cloud_text

BACKEND_NAME: Final = "jev"

# A tool roster can run to hundreds of entries on an MCP-heavy client. Names are
# identifiers the caller chose, not user prose, so they are not redacted - but
# they are capped so they cannot crowd the actual message out of the state.
_MAX_TOOL_NAMES: Final = 32
_MAX_TOOL_NAME_CHARS: Final = 64
# Only the most recent attempts carry signal about going in circles.
_MAX_EARLIER_ATTEMPTS: Final = 3
# Ceiling on any server-supplied fragment quoted into an error message.
_MAX_DETAIL_CHARS: Final = 40

_ErrorKind = Literal["timeout", "connection", "auth", "rate_limit", "server", "invalid", "skipped"]

# Built once at import: the questions are constant, and constructing them from
# DOMAINS/STEP_OUTCOMES rather than from the criteria dicts means a label added
# to `schema` without a criterion fails loudly here instead of shipping an
# undescribed option to the model.
_TASK_QUESTIONS: Final[dict[str, Question]] = {
    "domain": Choice(
        instructions=DOMAIN_INSTRUCTIONS,
        criteria={name: DOMAIN_CRITERIA[name] for name in DOMAINS},
    ),
    "difficulty": Score(
        instructions=DIFFICULTY_INSTRUCTIONS,
        criteria=list(DIFFICULTY_CRITERIA),
    ),
    "tool_need": Noul(
        instructions=TOOL_NEED_INSTRUCTIONS,
        criteria={"true": TOOL_NEED_CRITERIA["true"], "false": TOOL_NEED_CRITERIA["false"]},
    ),
    "ambiguity": Noul(
        instructions=AMBIGUITY_INSTRUCTIONS,
        criteria={"true": AMBIGUITY_CRITERIA["true"], "false": AMBIGUITY_CRITERIA["false"]},
    ),
}

_STEP_QUESTIONS: Final[dict[str, Question]] = {
    "outcome": Choice(
        instructions=STEP_OUTCOME_INSTRUCTIONS,
        criteria={name: STEP_OUTCOME_CRITERIA[name] for name in STEP_OUTCOMES},
    ),
    "progress": Noul(
        instructions=STEP_PROGRESS_INSTRUCTIONS,
        criteria={"true": STEP_PROGRESS_CRITERIA["true"], "false": STEP_PROGRESS_CRITERIA["false"]},
    ),
    "complete": Noul(
        instructions=STEP_COMPLETE_INSTRUCTIONS,
        criteria={"true": STEP_COMPLETE_CRITERIA["true"], "false": STEP_COMPLETE_CRITERIA["false"]},
    ),
    "needs_new_approach": Noul(
        instructions=STEP_NEW_APPROACH_INSTRUCTIONS,
        criteria={
            "true": STEP_NEW_APPROACH_CRITERIA["true"],
            "false": STEP_NEW_APPROACH_CRITERIA["false"],
        },
    ),
}


class _Cleaner:
    """Runs every outbound string through the privacy rules, remembering the kinds.

    Three steps, in order, for every string that goes into the state:

    1. **Machine output stays local.** Tracebacks, code blocks, terminal runs,
       key material and whole JSON/XML documents are replaced by a one-line
       description ("[Python traceback kept local: 23 lines]"). This is not
       switchable: the rule is that raw dumps never leave the machine, and the
       judgments only need to know such content is *there*, not what it says.
    2. **Secrets are redacted** from what remains (``privacy.redact``).
    3. **The whole state shares one character budget** (``privacy.max_state_chars``),
       spent in the order fields are cleaned - the current message first.

    One instance per call, so the notes it produces describe that request only.
    """

    __slots__ = ("_max_chars", "_redact", "kinds", "kept_local")

    def __init__(self, privacy: PrivacyConfig) -> None:
        self._max_chars = privacy.max_state_chars
        self._redact = privacy.redact
        self.kinds: list[str] = []
        self.kept_local: list[str] = []

    def __call__(self, text: str) -> str:
        minimized, kept = minimize_for_cloud(text)
        for kind in kept:
            if kind not in self.kept_local:
                self.kept_local.append(kind)
        prepared, kinds = prepare_cloud_text(
            minimized, max_chars=self._max_chars, do_redact=self._redact
        )
        self._max_chars = max(0, self._max_chars - len(prepared))
        for kind in kinds:
            if kind not in self.kinds:
                self.kinds.append(kind)
        return prepared

    def notes(self) -> list[str]:
        """Storable records of what was withheld - kinds only, never the content."""
        notes: list[str] = []
        if self.kept_local:
            notes.append(f"kept local: {', '.join(self.kept_local)}")
        if self.kinds:
            notes.append(f"redacted: {', '.join(self.kinds)}")
        return notes


def _tool_names(names: list[str]) -> list[str]:
    cleaned: list[str] = []
    for raw in names[:_MAX_TOOL_NAMES]:
        name = raw.strip()[:_MAX_TOOL_NAME_CHARS]
        if name:
            cleaned.append(name)
    return cleaned


def _task_state(context: TaskContext, clean: _Cleaner) -> dict[str, Any]:
    """The `request.*` object the task questions refer to.

    `system_prompt`, `session_id` and `estimated_prompt_tokens` are not sent: no
    question names them, a system prompt is the likeliest place for a secret to
    hide, and unnamed context only dilutes the judgment.
    """
    request: dict[str, Any] = {
        "message": clean(context.message),
        "previous_message": clean(context.prior_message) if context.prior_message else None,
        "available_tools": [clean(name) for name in _tool_names(context.tool_names)] or None,
        "turn": context.turn_index,
        "has_images": context.has_images,
    }
    return {"request": {key: value for key, value in request.items() if value is not None}}


def _step_state(context: StepContext, clean: _Cleaner) -> dict[str, Any]:
    """The `task.*` / `step.*` / `earlier_attempts` object the step questions refer to."""
    goal = clean(context.goal)
    # STEP_OUTCOME_INSTRUCTIONS names `step.goal` while the other step questions
    # name `task.goal`. That text is shared with the local and rules backends and
    # is not ours to rewrite, so the goal is mirrored into both paths: every path
    # a question names then resolves, at the cost of one repeated short string.
    step: dict[str, Any] = {
        "goal": goal,
        "action": clean(context.action),
        "result": clean(context.result),
        "exit_code": context.exit_code,
        "attempt": context.attempt,
    }
    state: dict[str, Any] = {
        "task": {"goal": goal},
        "step": {key: value for key, value in step.items() if value is not None},
    }
    earlier = [clean(item) for item in context.previous_attempts[-_MAX_EARLIER_ATTEMPTS:] if item]
    if earlier:
        state["earlier_attempts"] = earlier
    return state


def _classify(error: BaseException) -> tuple[_ErrorKind, bool]:
    """Map an SDK failure onto the breaker's vocabulary.

    Order matters: `TypeSafeAPITimeoutError` is a connection error and
    `TypeSafeAPIResponseValidationError` is an API error, so the specific cases
    are tested before the general ones.
    """
    if isinstance(error, TypeSafeAPITimeoutError):
        return "timeout", True
    if isinstance(error, TypeSafeAPIConnectionError):
        return "connection", True
    if isinstance(error, TypeSafeAPIResponseValidationError):
        # A 200 whose body we could not parse: retrying would return the same body.
        return "invalid", False
    if isinstance(error, TypeSafeAuthenticationError):
        return "auth", False
    if isinstance(error, TypeSafeRateLimitError):
        return "rate_limit", True
    if isinstance(error, TypeSafeUnprocessableEntityError):
        return "invalid", False
    if isinstance(error, TypeSafeInternalServerError):
        return "server", True
    if isinstance(error, TypeSafeAPIError):
        status = error.status
        if status in (401, 403):
            return "auth", False
        if status == 408:
            return "timeout", True
        if status == 429:
            return "rate_limit", True
        if 400 <= status < 500:
            return "invalid", False
        return "server", True
    if isinstance(error, httpx2.TimeoutException):
        return "timeout", True
    if isinstance(error, httpx2.HTTPError):
        return "connection", True
    # A plain TypeSafeError means the request we built was unusable (bad key, bad
    # timeout, empty questions). Retrying it unchanged cannot help.
    return "invalid", False


def _describe(error: BaseException) -> str:
    """A log-safe description: provider metadata only, never the response body.

    A server error body could in principle echo the state back, and these strings
    end up in logs and in the decision store.
    """
    if isinstance(error, TypeSafeAPIError):
        detail = f"{type(error).__name__} status={error.status}"
        request_id = error.request_id
        return f"{detail} request={request_id}" if request_id else detail
    return type(error).__name__


def _request_id(response: SystemOneResponse) -> str | None:
    try:
        return response.request_id
    except TypeSafeError:
        return None


class JevBackend:
    """Decision backend backed by the TypeSafe System One API.

    `facts` is accepted to satisfy `DecisionBackend` but never read: whether the
    local llama-server is up or busy has no bearing on a cloud call, which is
    precisely why this backend is first in the chain.
    """

    name: str = BACKEND_NAME

    def __init__(
        self,
        config: Config,
        *,
        client: AsyncTypeSafeClient | None = None,
        transport: httpx2.AsyncBaseTransport | None = None,
    ) -> None:
        """Hold the config; the HTTP client is built on first use.

        `transport` is the seam tests drive with `httpx2.MockTransport`; `client`
        replaces the whole SDK client. A caller-supplied client is the caller's to
        close, matching how `upstream.LlamaClient` treats an injected client.
        """
        if client is not None and transport is not None:
            raise ValueError("client and transport are mutually exclusive.")
        self._config = config
        self._transport = transport
        self._client = client
        self._owns_client = client is None
        self._closed = False

    # --- availability -------------------------------------------------------

    def _unconfigured_reason(self) -> str | None:
        """Why this backend must not be called, or None when it may be."""
        jev = self._config.decisions.jev
        # "auto" means "on when a key exists", which the key check below covers.
        if jev.enabled != "auto" and not jev.enabled:
            return "decisions.jev.enabled is false"
        if not self._config.privacy.allow_cloud:
            return "privacy.allow_cloud is false"
        if not self._config.typesafe_api_key:
            return "TYPESAFE_API_KEY is not set"
        return None

    async def availability(self, facts: BackendFacts) -> Availability:
        """Configuration only - this never opens a socket, so the chain can always ask."""
        if self._closed:
            return Availability.unconfigured("backend is closed")
        reason = self._unconfigured_reason()
        return Availability.ok() if reason is None else Availability.unconfigured(reason)

    # --- judgments ----------------------------------------------------------

    async def decide_task(self, context: TaskContext, facts: BackendFacts) -> TaskDecision:
        """One call, four questions: domain, difficulty, tool need, ambiguity."""
        watch = Stopwatch()
        clean = _Cleaner(self._config.privacy)
        response = await self._ask(_task_state(context, clean), _TASK_QUESTIONS)

        domain = self._choice(response, "domain", DOMAINS)
        difficulty = self._score(response, "difficulty", len(DIFFICULTY_CRITERIA))
        return TaskDecision(
            domain=cast(Domain, domain.choice),
            domain_probabilities=dict(domain.probabilities),
            domain_confidence=domain.confidence,
            # `.score` is already the probability-weighted level, so it is used
            # as returned rather than recomputed from `.probabilities`.
            difficulty=difficulty.score,
            # TaskDecision keys difficulty probabilities by string; the SDK gives
            # integer score levels.
            difficulty_probabilities={
                str(level): value for level, value in difficulty.probabilities.items()
            },
            difficulty_confidence=difficulty.confidence,
            tool_need=self._noul(response, "tool_need"),
            ambiguity=self._noul(response, "ambiguity"),
            backend=self.name,
            latency_ms=watch.ms,
            usage=self._usage(response.usage),
            notes=self._notes(response, clean),
        )

    async def decide_step(self, context: StepContext, facts: BackendFacts) -> StepDecision:
        """One call, four questions: outcome, progress, completeness, new approach."""
        watch = Stopwatch()
        clean = _Cleaner(self._config.privacy)
        response = await self._ask(_step_state(context, clean), _STEP_QUESTIONS)

        outcome = self._choice(response, "outcome", STEP_OUTCOMES)
        return StepDecision(
            outcome=cast(StepOutcome, outcome.choice),
            outcome_probabilities=dict(outcome.probabilities),
            outcome_confidence=outcome.confidence,
            progress=self._noul(response, "progress"),
            complete=self._noul(response, "complete"),
            needs_new_approach=self._noul(response, "needs_new_approach"),
            backend=self.name,
            latency_ms=watch.ms,
            usage=self._usage(response.usage),
            notes=self._notes(response, clean),
        )

    # --- transport ----------------------------------------------------------

    def _ensure_client(self) -> AsyncTypeSafeClient:
        if self._closed:
            raise self._error("backend is closed", kind="skipped", retryable=False)
        reason = self._unconfigured_reason()
        if reason is not None:
            # Refusing to run is not a provider failure, so it must not open the
            # breaker: `skipped` is the kind base.py reserves for exactly this.
            raise self._error(reason, kind="skipped", retryable=False)
        if self._client is None:
            jev = self._config.decisions.jev
            try:
                self._client = AsyncTypeSafeClient(
                    api_key=self._config.typesafe_api_key,
                    model=jev.model,
                    retry=RetryPolicy(
                        max_retries=max(0, jev.max_retries),
                        # Cap the whole retry sequence, not just one attempt, so a
                        # flapping provider cannot outlive the decision budget by
                        # an unbounded margin. decisions.budget_ms stays the
                        # chain's to enforce.
                        timeout=jev.timeout_s * (max(0, jev.max_retries) + 1),
                    ),
                    timeout=jev.timeout_s,
                    transport=self._transport,
                )
            except TypeSafeError as error:
                raise self._translate(error) from error
        return self._client

    async def _ask(self, state: dict[str, Any], questions: dict[str, Question]) -> SystemOneResponse:
        client = self._ensure_client()
        try:
            return await client.system_one(state, questions)
        except (TypeSafeError, httpx2.HTTPError) as error:
            raise self._translate(error) from error

    async def aclose(self) -> None:
        """Close the lazily built client. Idempotent; later calls are `skipped`."""
        self._closed = True
        client, self._client = self._client, None
        if client is not None and self._owns_client:
            await client.aclose()

    # --- answer parsing -----------------------------------------------------

    def _choice(
        self, response: SystemOneResponse, name: str, allowed: tuple[str, ...]
    ) -> ChoiceAnswer:
        answer = response.choices.get(name)
        if answer is None:
            raise self._missing(response, name, "choice")
        if answer.choice not in allowed:
            raise self._error(
                f"{name}: unknown label {answer.choice[:_MAX_DETAIL_CHARS]!r}",
                kind="invalid",
                retryable=False,
            )
        return answer

    def _score(self, response: SystemOneResponse, name: str, levels: int) -> ScoreAnswer:
        answer = response.scores.get(name)
        if answer is None:
            raise self._missing(response, name, "score")
        if not 0.0 <= answer.score <= levels - 1:
            raise self._error(
                f"{name}: score {answer.score} outside 0..{levels - 1}",
                kind="invalid",
                retryable=False,
            )
        return answer

    def _noul(self, response: SystemOneResponse, name: str) -> float:
        answer = response.nouls.get(name)
        if answer is None:
            raise self._missing(response, name, "noul")
        if not 0.0 <= answer.noul <= 1.0:
            raise self._error(
                f"{name}: probability {answer.noul} outside 0..1", kind="invalid", retryable=False
            )
        return answer.noul

    def _missing(self, response: SystemOneResponse, name: str, expected: str) -> BackendError:
        """A question that came back absent or as the wrong answer type.

        Filling it with a default would hand the policy a judgment nobody made,
        so this is a hard failure even though the other three answers parsed.
        """
        present = "absent" if name not in response.answers else "wrong type"
        return self._error(f"{name}: {expected} answer {present}", kind="invalid", retryable=False)

    # --- decision metadata --------------------------------------------------

    def _usage(self, usage: Usage) -> dict[str, int]:
        """Token counts plus an estimated price in micro-dollars.

        Only input tokens are billed, and `input_tokens * usd_per_input_mtok` is
        already the price in millionths of a dollar - an integer unit that keeps
        a single routing decision (tens of micro-dollars) from rounding to zero
        the way a USD float would. Keys are omitted when the API reported no
        usage rather than guessed at.
        """
        counts: dict[str, int] = {}
        if usage.input_tokens is not None:
            counts["input_tokens"] = usage.input_tokens
        if usage.output_tokens is not None:
            counts["output_tokens"] = usage.output_tokens
        if usage.input_tokens is not None:
            rate = self._config.decisions.jev.usd_per_input_mtok
            counts["est_cost_usd_micro"] = round(usage.input_tokens * rate)
        return counts

    def _notes(self, response: SystemOneResponse, clean: _Cleaner) -> list[str]:
        """Provenance for the store. Every entry is metadata; none is request text."""
        notes = [f"model: {response.model}"]
        request_id = _request_id(response)
        if request_id:
            notes.append(f"request: {request_id}")
        notes.extend(clean.notes())
        return notes

    # --- errors -------------------------------------------------------------

    def _error(self, detail: str, *, kind: _ErrorKind, retryable: bool) -> BackendError:
        return BackendError(
            f"{self.name}: {detail}", backend=self.name, kind=kind, retryable=retryable
        )

    def _translate(self, error: BaseException) -> BackendError:
        kind, retryable = _classify(error)
        return self._error(_describe(error), kind=kind, retryable=retryable)
