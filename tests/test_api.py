"""The supervisory API.

What an agent gets when it asks rather than chats. The property that matters
most here is that asking is free of side effects: ``/agent/route`` reports the
model it would use and must not restart anything unless explicitly told to.
"""

from __future__ import annotations

from typing import AsyncIterator

import pytest

from switchboard.config import Config

from conftest import DEEP, FAST, FakeExecutor, FakeLlama
from test_proxy import Harness


@pytest.fixture
async def harness(config: Config) -> AsyncIterator[Harness]:
    llama = FakeLlama(resident=FAST)
    async with Harness(config, llama, FakeExecutor()) as ready:
        yield ready


async def test_health_is_cheap_and_informative(harness: Harness) -> None:
    response = await harness.client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["mode"] == "active"
    assert payload["resident_model"] == "fast-9b"
    assert payload["upstream_healthy"] is True


async def test_status_reports_the_whole_stack(harness: Harness) -> None:
    payload = (await harness.client.get("/status")).json()

    assert payload["upstream"]["healthy"] is True
    assert payload["executor"]["name"] == "fake"
    assert len(payload["models"]) == 3
    assert "rules" in payload["providers"]
    assert "ram_free_gb" in payload["host"]


async def test_registry_exposes_measurements_and_problems(harness: Harness) -> None:
    payload = (await harness.client.get("/registry")).json()

    assert payload["resident"] == "fast-9b"
    models = {model["id"]: model for model in payload["models"]}
    assert models["med-27b"]["supports_tools"] is False
    assert models["fast-9b"]["gen_tps"]["value"] == 30.0
    assert models["fast-9b"]["context_limit"] == 114688


async def test_providers_can_be_probed(harness: Harness) -> None:
    payload = (await harness.client.get("/providers", params={"probe": True})).json()

    assert payload["order"] == ["rules"]
    assert payload["health"]["rules"]["availability"] == "ready"


async def test_route_is_side_effect_free_by_default(harness: Harness) -> None:
    response = await harness.client.post(
        "/agent/route",
        json={
            "message": "Work through this proof carefully and justify every step of the argument.",
            "session": "agent-1",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["selected_model"] in {"fast-9b", "deep-27b", "med-27b"}
    assert payload["reason"]
    assert payload["decision"]["backend"] == "rules"
    # Nothing was restarted just because something was asked.
    assert harness.executor.calls == []


async def test_route_with_apply_may_switch(config: Config) -> None:
    llama = FakeLlama(resident=FAST)
    executor = FakeExecutor()
    executor.on_switch = lambda model_id: setattr(llama, "resident", DEEP)
    async with Harness(config, llama, executor) as harness:
        response = await harness.client.post(
            "/agent/route",
            json={
                "message": "What is the differential diagnosis for exertional chest pain "
                "radiating to the jaw, and what is the immediate priority?",
                "apply": True,
                "policy": "medical",
            },
        )
        assert response.status_code == 200
        assert executor.calls, "apply=true must be allowed to load a model"


async def test_route_reports_what_it_would_have_done(harness: Harness) -> None:
    payload = (
        await harness.client.post(
            "/agent/route",
            json={"message": "Design a distributed scheduler and justify every trade-off.",
                  "policy": "deep"},
        )
    ).json()

    routing = payload["routing"]
    assert routing["would_switch_to"] or routing["selected"]
    assert routing["candidates"], "the arithmetic must be visible to the caller"


async def test_step_evaluation_returns_a_recommendation(harness: Harness) -> None:
    response = await harness.client.post(
        "/agent/step",
        json={
            "goal": "install the dependencies",
            "action": "uv sync",
            "result": "error: No such file or directory (os error 2)",
            "exit_code": 1,
            "attempt": 1,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["recommendation"] in {"retry", "retry_different"}
    assert payload["rationale"]
    assert payload["decision"]["outcome"] == "failure"


async def test_switch_rejects_an_unknown_model(harness: Harness) -> None:
    response = await harness.client.post("/agent/switch", json={"model": "llama-400b"})
    assert response.status_code == 404


async def test_switch_reports_a_refusing_executor(config: Config) -> None:
    llama = FakeLlama(resident=FAST)
    async with Harness(config, llama, FakeExecutor(ok=False)) as harness:
        response = await harness.client.post("/agent/switch", json={"model": "deep-27b"})
        assert response.status_code == 503
        assert response.json()["detail"]["error"]


async def test_routes_and_stats_work_on_an_empty_database(harness: Harness) -> None:
    routes = (await harness.client.get("/routes")).json()
    assert routes["routes"] == []

    stats = (await harness.client.get("/stats")).json()
    assert stats["routes"]["routes"] == 0
    assert stats["providers"] == {}


async def test_routes_show_up_after_a_decision(harness: Harness) -> None:
    await harness.client.post("/agent/route", json={"message": "hello"})
    routes = (await harness.client.get("/routes", params={"limit": 5})).json()["routes"]

    assert len(routes) == 1
    assert routes[0]["source"] == "api"
    assert routes[0]["selected_model"]


async def test_openapi_documents_the_agent_surface(harness: Harness) -> None:
    schema = (await harness.client.get("/openapi.json")).json()
    paths = set(schema["paths"])

    assert {"/agent/route", "/agent/step", "/agent/switch", "/status", "/registry"} <= paths
    route_post = schema["paths"]["/agent/route"]["post"]
    assert route_post["summary"]
