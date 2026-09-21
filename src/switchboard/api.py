"""The supervisory API.

The proxy serves chat clients; this serves agents. An agent loop does not want
a completion, it wants an answer to "which model should this task go to?" or
"did that step work, and what now?" - questions Switchboard can answer in one
round trip without generating anything.

Everything here is typed, so the generated OpenAPI documentation at ``/docs``
is the reference. Two conventions are worth stating:

* **Deciding is free of side effects by default.** ``/agent/route`` reports the
  model it would use; pass ``apply=true`` to let it actually load one.
* **Judgments and facts stay separable in the response.** ``decision`` is what
  the semantic backend said, ``routing`` is what the policy did with it, and
  ``recommendation`` is the action code derived from both. An agent that
  disagrees with the recommendation can still read the parts.
"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from switchboard.decisions.base import StepContext, TaskContext
from switchboard.policy import POLICIES, Policy
from switchboard.privacy import hash_text
from switchboard.router import Router


class RouteTaskRequest(BaseModel):
    """Ask which local model should handle a piece of work."""

    message: str = Field(description="The task or user request to route.")
    prior_message: str | None = Field(
        default=None, description="The previous user turn, if this is a continuing conversation."
    )
    system_prompt: str | None = Field(
        default=None, description="System prompt in force, if any. Used only for context sizing."
    )
    tools: list[str] = Field(
        default_factory=list,
        description="Names of tools available for this task. Models whose chat template "
        "cannot express tool calls are excluded when this is non-empty.",
    )
    session: str | None = Field(
        default=None,
        description="Conversation identifier. Hashed before storage; drives model stickiness.",
    )
    policy: Policy = Field(default="auto", description=f"One of {', '.join(POLICIES)}.")
    max_tokens: int | None = Field(default=None, description="Expected answer length, if known.")
    estimated_prompt_tokens: int | None = Field(
        default=None,
        description="Override the prompt size estimate (defaults to a character-based estimate).",
    )
    apply: bool = Field(
        default=False,
        description="When true, actually switch models if the policy calls for it. "
        "When false (the default) the call only reports what it would do.",
    )


class StepRequest(BaseModel):
    """Ask whether a finished agent step worked and what to do next."""

    goal: str = Field(description="What the overall task is trying to achieve.")
    action: str = Field(description="What this step did: the command, tool call or plan.")
    result: str = Field(description="What came back: output, error text, or a summary of it.")
    exit_code: int | None = Field(
        default=None, description="Process exit code, when the step ran a command."
    )
    attempt: int = Field(default=1, ge=1, description="Which attempt at this step this was.")
    previous_attempts: list[str] = Field(
        default_factory=list, description="Earlier actions tried for this same step."
    )
    elapsed_s: float | None = Field(default=None, description="How long the step took.")
    session: str | None = Field(default=None, description="Conversation identifier.")


class SwitchRequest(BaseModel):
    model: str = Field(description="Model id, alias or GGUF filename to load.")
    reason: str | None = Field(default=None, description="Recorded with the switch.")


def build_api_router(get_router) -> APIRouter:
    api = APIRouter()

    @api.get("/health", summary="Liveness and a one-line view of the stack")
    async def health(request: Request) -> dict[str, Any]:
        router: Router = get_router(request)
        status, resident = await router.facts()
        return {
            "status": "ok",
            "mode": router.config.routing.mode,
            "upstream_healthy": status.healthy,
            "resident_model": resident.id if resident else status.model_id,
            "inflight": router.inflight_count,
            "uptime_s": round(router.uptime_s, 1),
        }

    @api.get("/status", summary="Full status: upstream, models, providers, host")
    async def status(request: Request) -> dict[str, Any]:
        router: Router = get_router(request)
        return await router.status_report()

    @api.get("/providers", summary="Decision backend availability and breaker state")
    async def providers(request: Request, probe: bool = False) -> dict[str, Any]:
        router: Router = get_router(request)
        upstream_status, resident = await router.facts()
        facts = router.backend_facts(upstream_status, resident)
        health = await router.chain.probe(facts) if probe else router.chain.health()
        stats: dict[str, Any] = {}
        if router.store is not None:
            try:
                stats = router.store.provider_stats(hours=24)
            except Exception:  # noqa: BLE001
                stats = {}
        return {
            "order": [backend.name for backend in router.chain.backends],
            "health": health,
            "last_24h": stats,
        }

    # Deliberately not "/models": that path belongs to the OpenAI-compatible
    # proxy, which serves it with and without the /v1 prefix. This is the richer
    # registry view - measurements, priors, context limits, problems.
    @api.get("/registry", summary="The model registry, with measurements and priors")
    async def registry(request: Request, refresh: bool = False) -> dict[str, Any]:
        router: Router = get_router(request)
        snapshot = router.registry(refresh=refresh)
        upstream_status, resident = await router.facts()
        return {
            "resident": resident.id if resident else None,
            "active_in_stack_config": snapshot.active_model_file,
            "models": [model.describe() for model in snapshot.models],
            "problems": snapshot.problems,
        }

    @api.post("/agent/route", summary="Choose a model for a task")
    async def route_task(request: Request, payload: RouteTaskRequest) -> dict[str, Any]:
        router: Router = get_router(request)
        chars_per_token = router.config.policy.chars_per_token
        estimated = payload.estimated_prompt_tokens
        if estimated is None:
            total = len(payload.message) + len(payload.prior_message or "") + len(
                payload.system_prompt or ""
            )
            estimated = int(total / max(1.0, chars_per_token))

        context = TaskContext(
            message=payload.message,
            prior_message=payload.prior_message,
            system_prompt=payload.system_prompt,
            tool_names=list(payload.tools),
            estimated_prompt_tokens=estimated,
            session_id=hash_text(payload.session) if payload.session else None,
        )
        plan = await router.route(
            context,
            policy=payload.policy,
            max_tokens=payload.max_tokens,
            streaming=False,
            source="api",
            allow_switch=payload.apply,
        )
        return plan.describe()

    @api.post("/agent/step", summary="Evaluate a finished step and recommend the next action")
    async def evaluate_step(request: Request, payload: StepRequest) -> dict[str, Any]:
        router: Router = get_router(request)
        context = StepContext(
            goal=payload.goal,
            action=payload.action,
            result=payload.result,
            exit_code=payload.exit_code,
            attempt=payload.attempt,
            previous_attempts=list(payload.previous_attempts),
            elapsed_s=payload.elapsed_s,
            session_id=hash_text(payload.session) if payload.session else None,
        )
        evaluation = await router.evaluate_step(context)
        return evaluation.describe()

    @api.post("/agent/switch", summary="Load a specific model")
    async def switch(request: Request, payload: SwitchRequest) -> dict[str, Any]:
        router: Router = get_router(request)
        target = router.registry().resolve(payload.model)
        if target is None:
            raise HTTPException(status_code=404, detail=f"unknown model {payload.model!r}")
        if not target.exists:
            raise HTTPException(
                status_code=409, detail=f"{target.filename} is not present on disk"
            )
        result = await router.switch_to(target, trigger="explicit")
        if not result.ok:
            raise HTTPException(status_code=503, detail=result.describe())
        return result.describe()

    @api.get("/routes", summary="Recent routing decisions")
    async def routes(request: Request, limit: int = 25) -> dict[str, Any]:
        router: Router = get_router(request)
        if router.store is None:
            return {"routes": [], "note": "no store configured"}
        return {"routes": router.store.recent_routes(limit=limit)}

    @api.get("/stats", summary="Aggregate routing and provider statistics")
    async def stats(request: Request, hours: float = 24) -> dict[str, Any]:
        router: Router = get_router(request)
        if router.store is None:
            return {"note": "no store configured"}
        return {
            "window_hours": hours,
            "routes": router.store.route_stats(hours=hours),
            "providers": router.store.provider_stats(hours=hours),
            "switches": router.store.recent_switches(limit=10),
        }

    return api
