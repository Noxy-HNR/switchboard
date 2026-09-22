"""The decision backend that asks the model already loaded on this machine.

Switchboard's cheapest honest opinion comes from the llama.cpp server it is
already proxying: no credential, no network hop, no per-token bill. This module
turns that server into a System One provider so the local answers arrive in the
same shape as the cloud ones and stay directly comparable.

Four rules shape everything here:

* **The questions are not ours.** They come from `schema`, byte for byte, so a
  local answer and a Jev answer are answers to the same question. Only the
  transport differs.
* **Nothing is redacted, everything is truncated.** The text never leaves this
  machine - it goes to a model that is already reading the user's whole
  conversation - so redacting it would buy no privacy while costing signal.
  Truncation is still applied, because prompt processing is the slow half of a
  local decision and the state budget bounds it.
* **Thinking is off.** The resident Qwen3.5 models reason by default, which
  turns a 300 ms decision into a 30 s one. Both llama.cpp switches are sent,
  because which one a build honours depends on the chat template in the GGUF.
* **Discrete means discrete.** In the default answer mode the model returns one
  label per question. The adapter will happily hand back a one-hot distribution
  and a confidence of 1.0; that is an artefact, not a measurement, so the
  probability dicts are left empty and confidence is left `None`. A local
  model's stated probabilities are not calibrated either way - see `logit` for
  the experiment that tries to get real ones.

`availability()` refuses loudly rather than quietly: `--parallel 1` means a
decision issued while a real generation is running simply queues behind it, and
a 5 tok/s partially-offloaded 27B costs more time than routing can win back.
Both refusals carry a sentence `switchboard doctor` can print verbatim.
"""

from __future__ import annotations

from typing import Any, Final

import httpx
import httpx2
from system_one_adapter import (
    AsyncSystemOneAdapterClient,
    Choice,
    Noul,
    Score,
    SystemOneResponse,
)
from system_one_adapter.providers import Message, ProviderResult, translating
from typesafe_sdk import (
    Answer,
    ChoiceAnswer,
    NoulAnswer,
    Questions,
    ScoreAnswer,
    TypeSafeAPIConnectionError,
    TypeSafeAPIError,
    TypeSafeAPIResponseValidationError,
    TypeSafeAPITimeoutError,
    TypeSafeAuthenticationError,
    TypeSafeBadRequestError,
    TypeSafeError,
    TypeSafeInternalServerError,
    TypeSafeNotFoundError,
    TypeSafePermissionDeniedError,
    TypeSafeRateLimitError,
    TypeSafeUnprocessableEntityError,
)

from switchboard.config import Config
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
from switchboard.decisions.state import build_step_state, build_task_state
from switchboard.privacy import truncate_middle

BACKEND_NAME: Final = "local"

# llama-server in single-model mode ignores the request's `model` field, but the
# OpenAI shape requires one and a router build needs the real id. Facts supply the
# real id when they have it; this is the placeholder for when they do not.
FALLBACK_MODEL_NAME: Final = "resident"

CHAT_COMPLETIONS_PATH: Final = "/v1/chat/completions"

DISCRETE_NOTE: Final = (
    "discrete answer mode: the local model returned one label per question, so no "
    "probabilities or confidence are reported (a one-hot distribution would be an "
    "artefact of the mode, not a measurement)"
)

_ERROR_MESSAGE_CHARS: Final = 300


def _truncate(text: str, limit: int) -> str:
    """The local backends' cleaning step: size only, nothing withheld.

    The state budget is the cloud backend's, reused here for a different reason:
    not exposure, but the prompt-processing seconds a long request costs on a
    partially offloaded model.
    """
    return truncate_middle(text, limit)


def _noul(instructions: str, criteria: dict[str, str]) -> Noul:
    return Noul(
        instructions=instructions,
        criteria={"true": criteria["true"], "false": criteria["false"]},
    )


# --- the questions ----------------------------------------------------------
# Built once from the shared constants. The adapter revalidates and dumps them on
# every call, so sharing the instances between calls is safe.

TASK_QUESTIONS: Final[Questions] = {
    "domain": Choice(instructions=DOMAIN_INSTRUCTIONS, criteria=dict(DOMAIN_CRITERIA)),
    "difficulty": Score(instructions=DIFFICULTY_INSTRUCTIONS, criteria=list(DIFFICULTY_CRITERIA)),
    "tool_need": _noul(TOOL_NEED_INSTRUCTIONS, TOOL_NEED_CRITERIA),
    "ambiguity": _noul(AMBIGUITY_INSTRUCTIONS, AMBIGUITY_CRITERIA),
}

STEP_QUESTIONS: Final[Questions] = {
    "outcome": Choice(
        instructions=STEP_OUTCOME_INSTRUCTIONS, criteria=dict(STEP_OUTCOME_CRITERIA)
    ),
    "progress": _noul(STEP_PROGRESS_INSTRUCTIONS, STEP_PROGRESS_CRITERIA),
    "complete": _noul(STEP_COMPLETE_INSTRUCTIONS, STEP_COMPLETE_CRITERIA),
    "needs_new_approach": _noul(STEP_NEW_APPROACH_INSTRUCTIONS, STEP_NEW_APPROACH_CRITERIA),
}


# --- availability, shared with the logit backend ----------------------------


def resident_stack_availability(
    facts: BackendFacts,
    *,
    base_url: str,
    skip_when_busy: bool,
    min_resident_gen_tps: float,
    work: str,
) -> Availability:
    """Whether the resident llama.cpp model can be asked a decision question now.

    Shared with `logit` so both backends refuse for the same reasons and in the
    same words: `switchboard doctor` prints these sentences unedited.

    An unknown `resident_gen_tps` never blocks. Refusing on a number nobody has
    measured yet would keep the backend off forever, because the measurement only
    happens once it runs.
    """
    if not facts.server_healthy:
        return Availability.unavailable(f"llama-server at {base_url} is not answering /health")
    if facts.server_busy and skip_when_busy:
        return Availability.unavailable(
            "llama-server is already generating for another request; with --parallel 1 "
            f"{work} would queue behind it instead of answering inside the budget"
        )
    gen_tps = facts.resident_gen_tps
    if min_resident_gen_tps > 0 and gen_tps is not None and gen_tps < min_resident_gen_tps:
        return Availability.unavailable(
            f"resident model {facts.resident_model_id or '(unknown)'} generates "
            f"{gen_tps:.1f} tok/s, under the {min_resident_gen_tps:.1f} tok/s floor: "
            f"{work} would cost more time than a better route can win back"
        )
    return Availability.ok()


# --- the provider -----------------------------------------------------------


class LlamaCppProvider:
    """An `AsyncProvider` that speaks llama.cpp's OpenAI-compatible chat endpoint.

    Written against `httpx` rather than the OpenAI SDK so Switchboard keeps one
    HTTP stack and one dependency list. The adapter only needs three things from a
    provider - a model name, one request, and an error translation - so the SDK
    would buy nothing but a second connection pool.

    The request body is pinned: temperature 0 because a routing decision must be
    reproducible, thinking disabled by both known switches, and `cache_prompt` on
    so a second question about the same conversation reuses the KV cache.
    """

    model_name: str

    def __init__(
        self,
        *,
        base_url: str,
        client: httpx.AsyncClient,
        timeout: httpx.Timeout,
        max_output_tokens: int,
        model_name: str = FALLBACK_MODEL_NAME,
    ) -> None:
        self.model_name = model_name
        self._url = f"{base_url.rstrip('/')}{CHAT_COMPLETIONS_PATH}"
        self._client = client
        self._timeout = timeout
        self._max_output_tokens = max_output_tokens

    def build_body(self, messages: list[Message], schema: dict[str, Any]) -> dict[str, Any]:
        """The exact JSON posted to llama.cpp, exposed so tests can pin it."""
        body: dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": message.role, "content": message.content} for message in messages],
            "temperature": 0.0,
            "max_tokens": self._max_output_tokens,
            "cache_prompt": True,
            # Two switches for one behaviour: older builds read the template kwarg,
            # newer ones read reasoning_effort, and sending both costs nothing.
            "chat_template_kwargs": {"enable_thinking": False},
            "reasoning_effort": "none",
        }
        # Always constrain sampling with the schema, whatever the adapter's mode.
        # The backend runs the adapter in *prompted* mode (structured_outputs=False)
        # because llama.cpp uses response_format only as a grammar and never shows
        # the model the schema: in native mode the questions and criteria, which
        # live in the schema's descriptions, would never reach the model at all.
        # Prompted mode makes the adapter put the schema in the prompt itself, so
        # its own trace shows exactly what the model saw; the grammar here then
        # guarantees the reply parses.
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "evaluation", "schema": schema, "strict": True},
        }
        return body

    async def request(
        self,
        messages: list[Message],
        *,
        schema: dict[str, Any],
        structured: bool,
    ) -> ProviderResult:
        """Perform one chat completion and return its payload and token usage.

        `structured` is the adapter's mode flag; the grammar is attached either
        way (see `build_body`), so it changes nothing here.
        """
        del structured
        body = self.build_body(messages, schema)
        # The timeout rides on the request itself, so expiry aborts the HTTP call
        # rather than leaving llama-server generating into a dropped future.
        with translating(self.translate_error):
            response = await self._client.post(self._url, json=body, timeout=self._timeout)
            response.raise_for_status()
            payload = response.json()
        return _provider_result(payload)

    def translate_error(self, error: Exception) -> TypeSafeError:
        """Map an httpx failure to the SDK error family the adapter expects."""
        if isinstance(error, TypeSafeError):
            return error
        # Timeouts are checked first: httpx models them as transport errors.
        if isinstance(error, httpx.TimeoutException):
            return TypeSafeAPITimeoutError(self._timeout.read or self._timeout.connect or 0.0)
        if isinstance(error, httpx.HTTPStatusError):
            response = error.response
            return _api_error(
                response.status_code,
                _error_body(response),
                httpx2.Headers(dict(response.headers)),
            )
        if isinstance(error, httpx.HTTPError):
            return TypeSafeAPIConnectionError(f"{type(error).__name__}: {error}")
        return TypeSafeError(f"{type(error).__name__}: {error}")


def _provider_result(payload: Any) -> ProviderResult:
    """Read one llama.cpp chat completion, refusing a shape we cannot trust."""
    choices = payload.get("choices") if isinstance(payload, dict) else None
    if not isinstance(choices, list) or not choices:
        raise TypeSafeError("llama-server returned no choices")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise TypeSafeError("llama-server returned an empty message content")
    # A reply cut off at max_tokens is incomplete JSON at best and a partial
    # answer set at worst, as the adapter's own OpenAI provider also refuses.
    # It is a response we cannot trust, not an outage: the same max_tokens would
    # cut it off again, so it must neither be retried nor trip the breaker.
    if isinstance(choices[0], dict) and choices[0].get("finish_reason") == "length":
        raise TypeSafeAPIResponseValidationError(
            200, {"finish_reason": "length"}, httpx2.Headers(), "choices[0].finish_reason"
        )
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}

    def count(key: str) -> int | None:
        value = usage.get(key)
        return int(value) if isinstance(value, (int, float)) else None

    # Unreported counts are None rather than 0, so a missing number is never
    # recorded as a measured zero.
    return ProviderResult(
        text=content,
        input_tokens=count("prompt_tokens"),
        output_tokens=count("completion_tokens"),
    )


_STATUS_ERRORS: Final[dict[int, type[TypeSafeAPIError]]] = {
    400: TypeSafeBadRequestError,
    401: TypeSafeAuthenticationError,
    403: TypeSafePermissionDeniedError,
    404: TypeSafeNotFoundError,
    422: TypeSafeUnprocessableEntityError,
    429: TypeSafeRateLimitError,
}


def _api_error(status: int, body: Any, headers: httpx2.Headers) -> TypeSafeAPIError:
    """The SDK error class for an HTTP status, mirroring the SDK's own mapping."""
    error_type = _STATUS_ERRORS.get(status)
    if error_type is None:
        error_type = TypeSafeInternalServerError if status >= 500 else TypeSafeAPIError
    return error_type(status, body, headers)


def _error_body(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return response.text[:_ERROR_MESSAGE_CHARS]


# --- the backend ------------------------------------------------------------


class LocalDecisionBackend:
    """Answer the `schema` questions on the model llama-server already has loaded."""

    name: str = BACKEND_NAME

    def __init__(self, config: Config, *, client: httpx.AsyncClient | None = None) -> None:
        self._config = config
        self._settings = config.decisions.local
        self._base_url = config.upstream.base_url.rstrip("/")
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            limits=httpx.Limits(max_keepalive_connections=2, max_connections=4)
        )
        self._provider = LlamaCppProvider(
            base_url=self._base_url,
            client=self._client,
            timeout=httpx.Timeout(
                self._settings.timeout_s, connect=config.upstream.connect_timeout_s
            ),
            max_output_tokens=self._settings.max_output_tokens,
        )
        # No retries anywhere: llama-server has one slot and the decision layer has
        # a hard millisecond budget, so a second attempt would spend the budget the
        # next backend needs. Structured outputs are grammar-constrained, so a
        # malformed-JSON correction round would be answering a problem we do not have.
        self._adapter = AsyncSystemOneAdapterClient(
            structured_outputs=False,
            llm_answer_mode=self._settings.answer_mode,
            normalize_probabilities=True,
            model=self._provider,
        )

    @property
    def discrete(self) -> bool:
        return self._settings.answer_mode == "discrete"

    async def availability(self, facts: BackendFacts) -> Availability:
        if not self._settings.enabled:
            return Availability.unconfigured(
                "decisions.local.enabled is false in config/switchboard.toml"
            )
        return resident_stack_availability(
            facts,
            base_url=self._base_url,
            skip_when_busy=self._settings.skip_when_busy,
            min_resident_gen_tps=self._settings.min_resident_gen_tps,
            work="a structured local decision",
        )

    async def decide_task(self, context: TaskContext, facts: BackendFacts) -> TaskDecision:
        watch = Stopwatch()
        response = await self._evaluate(self.task_state(context), TASK_QUESTIONS, facts)

        domain_answer = _choice_answer(response, "domain")
        difficulty_answer = _score_answer(response, "difficulty")
        decision = TaskDecision(
            domain=_as_domain(domain_answer.choice),
            difficulty=float(difficulty_answer.score),
            tool_need=float(_noul_answer(response, "tool_need").noul),
            ambiguity=float(_noul_answer(response, "ambiguity").noul),
            backend=self.name,
            latency_ms=watch.ms,
            usage=_usage(response),
            notes=self._notes(),
        )
        if not self.discrete:
            decision.domain_probabilities = dict(domain_answer.probabilities)
            decision.domain_confidence = float(domain_answer.confidence)
            decision.difficulty_probabilities = {
                str(level): probability
                for level, probability in difficulty_answer.probabilities.items()
            }
            decision.difficulty_confidence = float(difficulty_answer.confidence)
        return decision

    async def decide_step(self, context: StepContext, facts: BackendFacts) -> StepDecision:
        watch = Stopwatch()
        response = await self._evaluate(self.step_state(context), STEP_QUESTIONS, facts)

        outcome_answer = _choice_answer(response, "outcome")
        decision = StepDecision(
            outcome=_as_outcome(outcome_answer.choice),
            progress=float(_noul_answer(response, "progress").noul),
            complete=float(_noul_answer(response, "complete").noul),
            needs_new_approach=float(_noul_answer(response, "needs_new_approach").noul),
            backend=self.name,
            latency_ms=watch.ms,
            usage=_usage(response),
            notes=self._notes(),
        )
        if not self.discrete:
            decision.outcome_probabilities = dict(outcome_answer.probabilities)
            decision.outcome_confidence = float(outcome_answer.confidence)
        return decision

    async def aclose(self) -> None:
        await self._adapter.aclose()
        if self._owns_client:
            await self._client.aclose()

    # --- state ----------------------------------------------------------

    def task_state(self, context: TaskContext) -> dict[str, Any]:
        """The shared task state (`decisions.state`), truncated but never redacted.

        This text goes to the model the user is already talking to, on the
        machine the text is already on, so nothing is withheld - only cut to size.
        """
        return build_task_state(context, _truncate, self._config.privacy.max_state_chars)

    def step_state(self, context: StepContext) -> dict[str, Any]:
        """The shared step state; the result gets whatever the other fields leave."""
        return build_step_state(context, _truncate, self._config.privacy.max_state_chars)

    # --- plumbing -------------------------------------------------------

    def _notes(self) -> list[str]:
        return [DISCRETE_NOTE] if self.discrete else []

    async def _evaluate(
        self, state: dict[str, Any], questions: Questions, facts: BackendFacts
    ) -> SystemOneResponse:
        # Name the model llama-server actually has loaded, so a router build routes
        # to the resident one instead of loading something else mid-decision.
        self._provider.model_name = facts.resident_model_id or FALLBACK_MODEL_NAME
        try:
            return await self._adapter.system_one(state, questions)
        except BackendError:
            raise
        except (TypeSafeError, ValueError, RuntimeError) as error:
            raise _backend_error(error) from error


# --- answer mapping ---------------------------------------------------------


def _answer(response: SystemOneResponse, key: str) -> Answer:
    answer = response.answers.get(key)
    if answer is None:
        raise BackendError(
            f"local backend: the model did not answer {key!r}",
            backend=BACKEND_NAME,
            kind="invalid",
            retryable=False,
        )
    return answer


def _choice_answer(response: SystemOneResponse, key: str) -> ChoiceAnswer:
    answer = _answer(response, key)
    if not isinstance(answer, ChoiceAnswer):
        raise _wrong_answer_type(key, "choice", answer)
    return answer


def _score_answer(response: SystemOneResponse, key: str) -> ScoreAnswer:
    answer = _answer(response, key)
    if not isinstance(answer, ScoreAnswer):
        raise _wrong_answer_type(key, "score", answer)
    return answer


def _noul_answer(response: SystemOneResponse, key: str) -> NoulAnswer:
    answer = _answer(response, key)
    if not isinstance(answer, NoulAnswer):
        raise _wrong_answer_type(key, "noul", answer)
    return answer


def _wrong_answer_type(key: str, expected: str, answer: Answer) -> BackendError:
    return BackendError(
        f"local backend: {key!r} came back as {answer.type!r}, expected {expected!r}",
        backend=BACKEND_NAME,
        kind="invalid",
        retryable=False,
    )


def _usage(response: SystemOneResponse) -> dict[str, int]:
    """Token counts the adapter reported; counts it could not report are left out."""
    counts: dict[str, int] = {}
    for key, value in (
        ("input_tokens", response.usage.input_tokens_total),
        ("output_tokens", response.usage.output_tokens_total),
    ):
        if value is not None:
            counts[key] = int(value)
    return counts


def _as_domain(value: str) -> Domain:
    """Narrow a label to the `Domain` literal; the grammar cannot emit anything else."""
    for domain in DOMAINS:
        if domain == value:
            return domain
    return "general"


def _as_outcome(value: str) -> StepOutcome:
    for outcome in STEP_OUTCOMES:
        if outcome == value:
            return outcome
    return "unclear"


# --- failure classification -------------------------------------------------


def _backend_error(error: Exception) -> BackendError:
    """Classify a provider failure so the breaker punishes only real outages.

    A refusal we caused (a malformed request, a 404 path) must not count against
    llama-server's health, and a timeout must, because it means the machine is
    too busy to help right now.
    """
    kind: str
    retryable = True
    if isinstance(error, TypeSafeAPITimeoutError):
        kind = "timeout"
    elif isinstance(error, TypeSafeAPIConnectionError):
        kind = "connection"
    elif isinstance(error, (TypeSafeAuthenticationError, TypeSafePermissionDeniedError)):
        kind, retryable = "auth", False
    elif isinstance(error, TypeSafeRateLimitError):
        kind = "rate_limit"
    elif isinstance(
        error,
        (
            TypeSafeBadRequestError,
            TypeSafeUnprocessableEntityError,
            TypeSafeNotFoundError,
            TypeSafeAPIResponseValidationError,
            ValueError,
        ),
    ):
        kind, retryable = "invalid", False
    elif isinstance(error, RuntimeError):
        # The adapter was closed under us: a refusal to run, not an outage, so it
        # must not open the breaker on a server that is perfectly healthy.
        kind, retryable = "skipped", False
    else:
        kind = "server"
    # Only the exception's own message: the adapter hangs the full attempt trace,
    # raw prompt included, off the error, and none of that may reach a log row.
    return BackendError(
        f"local backend: {type(error).__name__}",
        backend=BACKEND_NAME,
        kind=kind,  # type: ignore[arg-type]
        retryable=retryable,
    )
