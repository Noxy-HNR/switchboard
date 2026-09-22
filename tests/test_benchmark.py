"""The benchmark tool.

Its whole job is to replace estimates with measurements, so the tests are about
provenance and refusal: a number must say where it came from, and when it
cannot be measured honestly there must be no number at all.
"""

from __future__ import annotations

import json

import pytest

from switchboard.config import Config
from switchboard.evals.benchmark import measure_generation, measure_switch, run_benchmark
from switchboard.store import Store

from conftest import DEEP, FAST, FakeExecutor, FakeLlama, build_router


async def close(router) -> None:
    await router.chain.aclose()
    await router.upstream.aclose()
    if router.store is not None:
        router.store.close()


def completion_with(*, timings: dict | None, tokens: int = 48) -> dict:
    payload = {
        "id": "chatcmpl-bench",
        "object": "chat.completion",
        "model": "fake",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "1\n2\n3"}, "finish_reason": "length"}
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": tokens, "total_tokens": 12 + tokens},
    }
    if timings is not None:
        payload["timings"] = timings
    return payload


async def test_generation_prefers_llama_cpp_timings(config: Config) -> None:
    llama = FakeLlama(resident=FAST)
    llama.completion = completion_with(
        timings={"predicted_per_second": 38.5, "prompt_per_second": 310.0}
    )
    router = build_router(config, llama)
    try:
        sample = await measure_generation(config, router, "fast-9b", tokens=48)
    finally:
        await close(router)

    assert sample["ok"] is True
    metrics = {item["metric"]: item for item in sample["metrics"]}
    assert metrics["gen_tps"]["value"] == pytest.approx(38.5)
    assert metrics["gen_tps"]["derived_from"] == "timings"
    assert sample["server_busy"] is False


async def test_generation_falls_back_to_the_wall_clock_and_says_so(config: Config) -> None:
    llama = FakeLlama(resident=FAST)
    llama.completion = completion_with(timings=None)
    router = build_router(config, llama)
    try:
        sample = await measure_generation(config, router, "fast-9b", tokens=48)
    finally:
        await close(router)

    metrics = {item["metric"]: item for item in sample["metrics"]}
    assert metrics["gen_tps"]["derived_from"] == "wall_clock"
    assert metrics["gen_tps"]["value"] > 0


async def test_a_busy_server_is_measured_but_flagged(config: Config) -> None:
    """A slow number with a known reason is evidence; a missing number is not."""
    llama = FakeLlama(resident=FAST, busy=True)
    llama.completion = completion_with(timings={"predicted_per_second": 6.0})
    router = build_router(config, llama)
    try:
        sample = await measure_generation(config, router, "fast-9b", tokens=48)
    finally:
        await close(router)

    assert sample["ok"] is True
    assert sample["server_busy"] is True


async def test_nothing_is_invented_when_llama_server_is_down(config: Config) -> None:
    llama = FakeLlama(healthy=False)
    router = build_router(config, llama)
    try:
        report = await run_benchmark(config, tokens=32)
    finally:
        await close(router)

    assert report["samples"] == []
    assert "not answering" in json.dumps(report).lower() or report.get("problem")


async def test_measurements_reach_the_store_for_the_registry(config: Config) -> None:
    llama = FakeLlama(resident=FAST)
    llama.completion = completion_with(timings={"predicted_per_second": 41.25})
    store = Store(config.db_file)
    router = build_router(config, llama, store=store)
    try:
        await run_benchmark(config, router=router, tokens=48)
        measurements = store.measurements_by_model()
    finally:
        await close(router)

    assert measurements["fast-9b"]["gen_tps"].value == pytest.approx(41.25, rel=0.01)
    assert measurements["fast-9b"]["gen_tps"].measured is True


async def test_a_switch_is_refused_while_the_server_is_busy(config: Config) -> None:
    """A switch is a process restart: never while it is generating for someone."""
    llama = FakeLlama(resident=FAST, busy=True)
    executor = FakeExecutor()
    router = build_router(config, llama, executor=executor)
    try:
        result = await measure_switch(config, router, "deep-27b")
    finally:
        await close(router)

    assert result["ok"] is False
    assert "generating" in (result["refused"] or "").lower()
    assert executor.calls == []


async def test_switches_are_not_measured_unless_asked(config: Config) -> None:
    llama = FakeLlama(resident=FAST)
    llama.completion = completion_with(timings={"predicted_per_second": 30.0})
    executor = FakeExecutor()
    router = build_router(config, llama, executor=executor)
    try:
        await run_benchmark(config, router=router, tokens=32)
    finally:
        await close(router)

    assert executor.calls == [], "the default run must not restart llama-server"
