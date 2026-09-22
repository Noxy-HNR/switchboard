"""Read-only cptr/config and real MCP compatibility checks. No credential output.

Run with cptr's existing Python installation to test its MCP client version.
"""
import asyncio
import importlib.metadata
import json
import sqlite3
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

ROOT = Path(__file__).resolve().parents[1]

def decode(value):
    return json.loads(value) if isinstance(value, str) else value

async def main():
    before = tomllib.loads((ROOT/'backups/cptr-config.toml.20260921-151843.bak').read_text(encoding='utf-8'))['app_config']
    current = tomllib.loads((Path.home()/'.cptr/config.toml').read_text(encoding='utf-8'))['app_config']
    changed = {'chat.connections', 'tool_servers'}
    checks = {'other_cptr_settings_preserved': all(current.get(k)==v for k,v in before.items() if k not in changed)}
    for key in sorted(changed):
        original, now = decode(before[key]), decode(current[key])
        checks[key+'_old_entries_preserved'] = all(x in now for x in original)
        checks[key+'_addition_present'] = any(x.get('name')=='Switchboard' for x in now)
    # cptr's supported TOML was loaded into its actual runtime configuration DB.
    path = Path.home()/'.cptr/app.db'
    with sqlite3.connect(path.as_uri()+'?mode=ro', uri=True) as db:
        for key in sorted(changed):
            row = db.execute('SELECT value FROM config WHERE key=?', (key,)).fetchone()
            checks[key+'_in_runtime_db'] = row is not None and decode(row[0]) == decode(current[key])
    async with httpx.AsyncClient(timeout=10) as client:
        checks['cptr_http'] = (await client.get('http://127.0.0.1:8000')).status_code==200
    async with streamablehttp_client('http://127.0.0.1:8002/mcp/') as (read, write, _):
        async with ClientSession(read, write) as session:
            info = await session.initialize()
            names = [t.name for t in (await session.list_tools()).tools]
            checks['mcp_tools'] = sorted(names)==['switchboard_evaluate_step','switchboard_route','switchboard_status']
            for name, args in [('switchboard_status',{}), ('switchboard_route',{'task':'hello'})]:
                checks[name] = not (await session.call_tool(name,args)).isError
    report = {'generated_at':datetime.now(timezone.utc).isoformat(), 'mcp_client':importlib.metadata.version('mcp'),
              'protocol':info.protocolVersion, 'checks':checks,
              'authenticated_ui':'BLOCKED_EXTERNAL: requires user login; no session fabricated'}
    (ROOT/'reports/integration-validation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    return int(not all(checks.values()))

if __name__ == '__main__':
    raise SystemExit(asyncio.run(main()))
