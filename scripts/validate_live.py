"""Reproducible, synthetic, 9B-only live acceptance checks; no model switches.

Run after scripts/start.ps1. Refuses any resident other than the approved 9B.
Reports contain measurements and check outcomes, never test prompt/response text.
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx
import httpx2
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from switchboard.config import load_config
from switchboard.decisions.base import BackendFacts, TaskContext
from switchboard.decisions.chain import DecisionChain
from switchboard.decisions.jev import JevBackend
from switchboard.decisions.local import LocalDecisionBackend
from switchboard.decisions.rules import RulesBackend

ROOT = Path(__file__).resolve().parents[1]
APP = 'http://127.0.0.1:8002'
DIRECT = 'http://127.0.0.1:10000'
MODEL = 'qwen3.5-9b'


async def main() -> int:
    checks: list[dict] = []
    def check(name, condition, **evidence):
        row = dict(name=name, status='PASS' if condition else 'FAIL', **evidence)
        checks.append(row)
        print(json.dumps(row), flush=True)

    cfg = load_config()
    marker = 'sb-private-' + uuid.uuid4().hex
    fake_secret = 'sk-live-' + uuid.uuid4().hex
    async with httpx.AsyncClient(timeout=90) as client:
        health = (await client.get(APP + '/health')).json()
        if health.get('resident_model') != MODEL or not health.get('upstream_healthy'):
            raise RuntimeError('Refusing live tests: approved 9B must already be healthy/resident')
        check('health', True, resident=health['resident_model'])
        data = (await client.get(APP + '/v1/models')).json()
        ids = [entry['id'] for entry in data['data']]
        check('models', 'switchboard-auto' in ids and MODEL in ids, models=ids)
        check('dashboard_and_openapi',
              (await client.get(APP + '/dashboard')).status_code == 200 and
              '/agent/step' in (await client.get(APP + '/openapi.json')).json()['paths'])

        async def completion(name, message, model='switchboard-auto', **extra):
            body = dict(model=model, messages=[dict(role='user', content=message)],
                        max_tokens=128, temperature=0, chat_template_kwargs={'enable_thinking': False})
            body.update(extra)
            start = time.perf_counter()
            response = await client.post(APP + '/v1/chat/completions', json=body,
                                         headers={'X-Switchboard-Session': 'live-validation'})
            elapsed = (time.perf_counter() - start) * 1000
            payload = response.json()
            good = response.status_code == 200 and bool(payload.get('choices'))
            check(name, good, http=response.status_code, total_ms=round(elapsed, 1),
                  backend=response.headers.get('x-switchboard-backend'),
                  model=response.headers.get('x-switchboard-model'), usage=payload.get('usage'))
            return response, payload

        await completion('easy_auto', 'What is two plus two? Reply briefly.')
        await completion('coding_auto', 'Write a Python function to add two integers, with one assertion test.')
        r, _ = await completion('explicit_model', 'Reply with the word ready.', model=MODEL)
        check('explicit_bypasses_semantics', r.headers.get('x-switchboard-backend') == 'bypass')
        r = await client.post(APP + '/v1/chat/completions', json={
            'model':'nonexistent-model', 'messages':[{'role':'user','content':'hello'}]})
        check('unknown_model_rejected', r.status_code == 404)

        route = (await client.post(APP + '/agent/route', json={
            'message': 'Design a lock-free concurrent work-stealing scheduler and prove its memory ordering invariants.',
            'apply':False})).json()
        check('hard_route_advice_only', route.get('selected_model') is not None and route.get('switch') is None,
              backend=route.get('decision', {}).get('backend'), decision=route.get('decision'),
              selected=route.get('selected_model'))
        for name, result, exit_code in [('step_success','5 tests passed; all requested tests passed',0),
                                        ('step_failure','Connection timed out; no file downloaded',1)]:
            step = (await client.post(APP + '/agent/step', json={
                'goal':'Run the five unit tests' if exit_code == 0 else 'Download a file',
                'action':'Run test suite' if exit_code == 0 else 'Download the file',
                'result':result, 'exit_code':exit_code})).json()
            check(name, step.get('recommendation') in (
                ['stop_complete','continue'] if exit_code == 0 else ['retry','retry_different','escalate','stop_failed']),
                recommendation=step.get('recommendation'), backend=step.get('decision',{}).get('backend'))

        function = {'type':'function','function':{'name':'add_numbers','description':'Add two integers.',
                    'parameters':{'type':'object','properties':{'a':{'type':'integer'},'b':{'type':'integer'}},
                                  'required':['a','b']}}}
        r, payload = await completion('tools_request', 'Use add_numbers with a=2 and b=3.', model=MODEL,
                        tools=[function], tool_choice={'type':'function','function':{'name':'add_numbers'}})
        calls = payload.get('choices',[{}])[0].get('message',{}).get('tool_calls',[])
        check('tool_definition_survived', bool(calls) and calls[0]['function']['name'] == 'add_numbers')

        started = time.perf_counter()
        arrivals = []
        done = False
        stream_error = False
        async with client.stream('POST', APP + '/v1/chat/completions', json={
            'model':MODEL, 'messages':[{'role':'user','content':'Count from 1 to 30, one number per line.'}],
            'stream':True, 'stream_options':{'include_usage':True}, 'max_tokens':100,
            'chat_template_kwargs':{'enable_thinking':False}}) as stream:
            async for line in stream.aiter_lines():
                if not line.startswith('data:'): continue
                content = line[5:].strip()
                if content == '[DONE]': done = True; continue
                obj = json.loads(content)
                stream_error |= 'error' in obj
                if any(c.get('delta',{}).get('content') for c in obj.get('choices',[])):
                    arrivals.append((time.perf_counter()-started)*1000)
        check('incremental_stream', done and not stream_error and len(arrivals)>3 and arrivals[-1]-arrivals[0]>100,
              chunks=len(arrivals), first_content_ms=round(arrivals[0],1) if arrivals else None,
              content_span_ms=round(arrivals[-1]-arrivals[0],1) if arrivals else None)

        await completion('privacy_canary_request', f'{marker} password="{fake_secret}". Say OK.', model=MODEL)

        # This mock never leaves the process; the fallback really calls the 9B.
        config = cfg.model_copy(deep=True)
        config.typesafe_api_key = 'synthetic-validation-key'
        config.decisions.jev.enabled = True
        config.decisions.jev.max_retries = 0
        config.decisions.fast_path.enabled = False
        transport = httpx2.MockTransport(lambda request: httpx2.Response(503, json={'error':'synthetic outage'}))
        chain = DecisionChain(config, [JevBackend(config, transport=transport),
                                      LocalDecisionBackend(config), RulesBackend(config)])
        try:
            fallback = await chain.decide_task(TaskContext(message='Write a Python binary search function.'),
                          BackendFacts(resident_model_id=MODEL, server_healthy=True), use_cache=False)
            check('jev_outage_fallback', fallback.backend == 'local' and fallback.fell_back,
                  backend=fallback.backend, attempts=fallback.describe()['attempts'])
        finally:
            await chain.aclose()

        async with streamablehttp_client(APP + '/mcp/') as (read, write, _):
            async with ClientSession(read, write) as mcp:
                await mcp.initialize()
                names = [tool.name for tool in (await mcp.list_tools()).tools]
                check('mcp_discovery', len(names)==3, tools=names)
                for name, args in [('switchboard_status', {}), ('switchboard_route', {'task':'Hello'}),
                    ('switchboard_evaluate_step', {'goal':'Run tests','action':'Run tests','result':'All tests passed','exit_code':0})]:
                    result = await mcp.call_tool(name, args)
                    check('mcp_' + name, not result.isError)

        direct = await client.post(DIRECT + '/v1/chat/completions', json={
            'model':MODEL, 'messages':[{'role':'user','content':'Say ready.'}], 'max_tokens':16,
            'chat_template_kwargs':{'enable_thinking':False}})
        check('direct_llama_still_works', direct.status_code == 200 and bool(direct.json().get('choices')))
        check('resident_unchanged', (await client.get(APP+'/health')).json()['resident_model'] == MODEL)

    # Inspect persistent observability only; fixtures and source deliberately
    # contain synthetic text. No user data or credential value is printed.
    with sqlite3.connect(cfg.db_file) as db:
        dump = '\n'.join(db.iterdump())
    logs = '\n'.join(p.read_text(encoding='utf-8', errors='replace') for p in (ROOT/'logs').glob('*.log'))
    check('privacy_canaries_not_persisted', all(c not in dump+logs for c in (marker, fake_secret)))
    report = {'generated_at':datetime.now(timezone.utc).isoformat(), 'checks':checks,
              'passed':sum(x['status']=='PASS' for x in checks), 'total':len(checks)}
    output = ROOT/'reports'
    output.mkdir(exist_ok=True)
    stem = 'live-' + datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
    (output/(stem+'.json')).write_text(json.dumps(report, indent=2), encoding='utf-8')
    (output/(stem+'.md')).write_text('# Live acceptance\n\n' + report['generated_at'] + '\n\n' +
        '\n'.join('- '+x['status']+': '+x['name'] for x in checks), encoding='utf-8')
    print('Report: '+str(output/stem), flush=True)
    return int(report['passed'] != report['total'])


if __name__ == '__main__':
    raise SystemExit(asyncio.run(main()))
