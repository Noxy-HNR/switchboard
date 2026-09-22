"""Regressions found by auditing real request, failure and switching paths."""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from switchboard.config import Config
from switchboard.decisions.base import TaskContext, BackendFacts, BackendError
from switchboard.privacy import redact, truncate_middle
from switchboard.proxy import _stream_response, _stream_with_keepalive
from conftest import DEEP, FAST, FakeExecutor, FakeLlama, build_router
from test_proxy import Harness, chat_body
from test_jev_backend import make_config, responding, TASK_BODY, body_of
from switchboard.decisions.jev import JevBackend
from switchboard.decisions.base import StepContext
from switchboard.decisions.chain import DecisionChain
from switchboard.decisions.rules import RulesBackend
from test_chain import Recorder, HARD, FACTS


async def test_cache_does_not_retain_raw_text_or_mutable_results(config):
    config.decisions.fast_path.enabled = False
    backend = Recorder()
    chain = DecisionChain(config, [backend])
    first = await chain.decide_task(HARD, FACTS)
    first.decision.notes.append('caller-only mutation')
    second = await chain.decide_task(HARD, FACTS)
    assert 'caller-only mutation' not in second.decision.notes
    assert HARD.message not in repr(chain._cache._entries)
    await chain.decide_task(HARD, BackendFacts(resident_model_id='another'))
    assert backend.calls == 2


async def test_step_cache_includes_deterministic_result_and_retry_history(config):
    backend = Recorder()
    chain = DecisionChain(config, [backend])
    for kwargs in [{}, {'exit_code': 1}, {'previous_attempts': ['retry']}, {'elapsed_s': 9}]:
        await chain.decide_step(StepContext(goal='g', action='a', result='r', **kwargs), FACTS)
    assert backend.calls == 4


async def test_availability_failure_is_bounded_private_and_falls_back(config):
    config.decisions.fast_path.enabled = False
    class Broken(Recorder):
        async def availability(self, facts):
            raise ValueError('raw prompt should not escape')
    chain = DecisionChain(config, [Broken(), RulesBackend(config)])
    result = await chain.decide_task(HARD, FACTS)
    assert result.backend == 'rules'
    assert 'raw prompt' not in str(result.describe()) + str(chain.health())


async def test_local_failure_does_not_persist_echoed_response():
    from test_local_backend import replying, backend_for, READY
    raw = 'synthetic private message echoed by upstream'
    async with backend_for(replying({'error': {'message': raw}}, status=500)) as backend:
        with pytest.raises(BackendError) as caught:
            await backend.decide_task(TaskContext(message=raw), READY)
    assert raw not in str(caught.value)


async def test_disconnected_switch_holds_lock_until_external_action_finishes(config):
    entered, finish = asyncio.Event(), asyncio.Event()
    executor = FakeExecutor()
    original = executor.switch
    async def delayed(*args, **kwargs):
        entered.set()
        await finish.wait()
        return await original(*args, **kwargs)
    executor.switch = delayed
    async with Harness(config, FakeLlama(), executor) as h:
        # switch_to takes a registry entry, not a filename: passing the string
        # made the task fail before it reached the executor, so `entered` was
        # never set and the test hung instead of failing.
        deep = h.router.registry().by_id("deep-27b")
        task = asyncio.create_task(h.router.switch_to(deep))
        await asyncio.wait_for(entered.wait(), timeout=5.0)
        task.cancel()
        await asyncio.sleep(0)
        assert h.router._switch_lock.locked()
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not h.router._switch_lock.locked()


@pytest.mark.parametrize('text,secret', [
    ('Authorization: Bearer abcdefghijklmnopqrst', 'abcdefghijklmnopqrst'),
    ('Authorization: Basic dXNlcjpwYXNz', 'dXNlcjpwYXNz'),
    ('{"password": "two word password"}', 'two word password'),
    ('{"api_key": "short"}', 'short'),
    ('password=abc', 'abc'),
    ('Cookie: session=abcdefg', 'abcdefg'),
])
def test_full_credentials_are_removed(text, secret):
    assert secret not in redact(text).text


@pytest.mark.parametrize('limit', [0, 1, 16, 33, 80, 1500])
def test_truncation_has_a_real_bound(limit):
    assert len(truncate_middle('z' * 20000, limit)) <= limit


async def test_cloud_never_receives_a_terminal_dump():
    """The dump stays local; the user's own question still goes.

    This test first asserted that the whole Jev call was refused whenever a
    message looked like a dump. That did keep dumps local, but the detector
    fired on any code fence or any message over eight lines, which kept most
    real coding questions away from Jev entirely. The requirement is that the
    *dump* never leaves, so that is now asserted directly on the bytes sent,
    along with the question that should still be judged.
    """
    transport, seen = responding(TASK_BODY)
    backend = JevBackend(make_config(), transport=transport)
    private = "private_frame_contents = load_secrets()"
    message = (
        "Why does this fail?\n"
        "Traceback (most recent call last):\n"
        '  File "app.py", line 3, in <module>\n'
        f"    {private}\n"
        "ValueError: private exception detail\n"
    )
    try:
        decision = await backend.decide_task(TaskContext(message=message), BackendFacts())
        sent = json.dumps(body_of(seen[0]))
        assert private not in sent
        assert "private exception detail" not in sent
        assert "Why does this fail?" in sent
        assert "Python traceback kept local" in sent
        assert any(note.startswith("kept local:") for note in decision.notes)
    finally:
        await backend.aclose()


@pytest.mark.parametrize("dump", [
    "```\nDB_PASSWORD=hunter2\nAPI_TOKEN=abc\n```",
    "OPENAI_API_KEY=sk-abcdef\nDB_HOST=db.internal\nDB_PASS=hunter2",
    'PS C:\\AI> Get-Content .env\nPS C:\\AI> type secrets.txt\nPS C:\\AI> dir',
    '{"user": "sam", "session_token": "abc123", "history": [1, 2, 3]}',
    "-----BEGIN OPENSSH PRIVATE KEY-----\nb3BlbnNzaC1rZXktdjEAAAAA\n-----END OPENSSH PRIVATE KEY-----",
])
async def test_no_kind_of_pasted_dump_reaches_the_cloud(dump):
    transport, seen = responding(TASK_BODY)
    backend = JevBackend(make_config(redact=False), transport=transport)
    try:
        # Redaction is off here on purpose: minimization must hold by itself.
        await backend.decide_task(TaskContext(message=f"Can you help with this?\n{dump}"), BackendFacts())
        sent = json.dumps(body_of(seen[0]))
        for fragment in ("hunter2", "abc123", "b3BlbnNzaC1rZXktdjEAAAAA", "secrets.txt", "sk-abcdef"):
            assert fragment not in sent
        assert "Can you help with this?" in sent
    finally:
        await backend.aclose()


async def test_cloud_text_budget_is_shared_across_fields():
    transport, seen = responding(TASK_BODY)
    backend = JevBackend(make_config(max_state_chars=200), transport=transport)
    try:
        await backend.decide_task(TaskContext(message='x'*5000, prior_message='y'*5000, tool_names=['z'*64]*30), BackendFacts())
        state = body_of(seen[0])['state']['request']
        assert sum(len(s) for s in [state['message'], state.get('previous_message', ''), *state.get('available_tools', [])]) <= 200
    finally:
        await backend.aclose()


async def test_explicit_failure_never_forwards_to_wrong_model(config):
    async with Harness(config, FakeLlama(), FakeExecutor(ok=False)) as h:
        r = await h.client.post('/v1/chat/completions', json=chat_body(model='deep-27b'))
        assert r.status_code == 409
        assert not h.llama.requests


async def test_explicit_model_does_not_call_a_decision_provider(config):
    async with Harness(config, FakeLlama(), FakeExecutor()) as h:
        async def forbidden(*args, **kwargs):
            raise AssertionError('explicit selection must bypass semantic providers')
        h.router.chain.decide_task = forbidden
        r = await h.client.post('/v1/chat/completions', json=chat_body(model='fast-9b'))
        assert r.status_code == 200
        assert r.headers['x-switchboard-backend'] == 'bypass'


async def test_browser_origin_and_request_memory_are_bounded(config):
    async with Harness(config, FakeLlama(), FakeExecutor()) as h:
        for origin in ['https://evil.example', 'null', 'http://localhost:8000', 'http://localhost:8002.evil.example']:
            r = await h.client.post('/agent/switch', json={'model':'deep-27b'}, headers={'Origin':origin})
            assert r.status_code == 403
        r = await h.client.post('/agent/switch', content='{"model":"deep-27b"}', headers={'Content-Type':'text/plain'})
        assert r.status_code == 415
        r = await h.client.post('/agent/switch', content=b'x' * (8*1024*1024+1), headers={'Content-Type':'application/json'})
        assert r.status_code == 413
        assert h.executor.calls == []


async def test_advice_does_not_advance_a_real_conversation(config):
    async with Harness(config, FakeLlama(), FakeExecutor()) as h:
        ctx = TaskContext(message='Hello', session_id='abc123')
        await h.router.route(ctx, allow_switch=False)
        assert h.router.store.get_session('abc123') is None


async def test_no_prompt_preview_by_default(config):
    assert Config().privacy.store_previews is False
    async with Harness(config, FakeLlama(), FakeExecutor()) as h:
        await h.client.post('/v1/chat/completions', json=chat_body())
        assert h.router.store.recent_routes()[0]['preview'] is None


async def test_explicit_switch_refuses_inflight_and_direct_busy_requests(config):
    llama, executor = FakeLlama(), FakeExecutor()
    async with Harness(config, llama, executor) as h:
        deep = h.router.registry().by_id('deep-27b')
        async with h.router.inflight():
            assert not (await h.router.switch_to(deep)).ok
        llama.busy = True
        assert not (await h.router.switch_to(deep)).ok
        assert executor.calls == []


async def test_generation_waits_for_switch_lock(config):
    async with Harness(config, FakeLlama(), FakeExecutor()) as h:
        entered = asyncio.Event()
        async def generation():
            async with h.router.inflight():
                entered.set()
        async with h.router._switch_lock:
            task = asyncio.create_task(generation())
            await asyncio.sleep(0)
            assert not entered.is_set()
        await task
        assert entered.is_set() and h.router.inflight_count == 0


async def test_stream_error_never_persists_an_upstream_echo(config):
    llama = FakeLlama(fail_with=500)
    async with Harness(config, llama, FakeExecutor()) as h:
        await h.client.post('/v1/chat/completions', json=chat_body(stream=True))
        row = h.router.store.recent_routes()[0]
        assert row['error'] == 'upstream returned HTTP 500'


async def test_model_drift_is_rejected_before_generation(config):
    llama = FakeLlama()
    async with Harness(config, llama, FakeExecutor()) as h:
        plan = await h.router.route(TaskContext(message='hello'))
        llama.resident = DEEP
        chunks = [c async for c in _stream_response(h.router, plan, chat_body(stream=True))]
        assert b'upstream_error' in b''.join(chunks)
        assert llama.requests == []


async def test_closing_keepalive_cancels_orphaned_routing(config, monkeypatch):
    import switchboard.proxy as proxy
    monkeypatch.setattr(proxy, '_KEEPALIVE_EVERY_S', 0.001)
    async with Harness(config, FakeLlama(), FakeExecutor()) as h:
        task = asyncio.create_task(asyncio.sleep(100))
        stream = _stream_with_keepalive(h.router, task, chat_body(stream=True), 'switchboard-auto', True)
        assert (await anext(stream)).startswith(b':')
        await stream.aclose()
        assert task.cancelled()
