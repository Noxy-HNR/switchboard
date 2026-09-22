# Integration

How Switchboard fits into the stack that was already here. Everything is
additive: every existing path still works with Switchboard stopped.

## cptr (done)

`scripts/integrate-cptr.ps1` added two things to `~/.cptr/config.toml`, after
backing it up:

* a chat connection named **Switchboard** at `http://127.0.0.1:8002`
  (`provider = openai`, `provider_type = llama.cpp`, no API key);
* an MCP tool server **switchboard** at `http://127.0.0.1:8002/mcp`.

The existing **Llama.cpp** connection to `:10000` and the existing **Personal
Health** MCP server were left exactly as they were.

cptr's own `db.py` says the TOML file wins at startup and "allows hand-editing",
which is why this path is used rather than the admin API - that needs a browser
session cookie a script cannot get. **cptr must be stopped when the script
runs**, because it rewrites the whole file from its database whenever a setting
changes.

### Using it

In cptr, pick the **Switchboard** connection and the model `switchboard-auto`.
That is automatic routing. The other virtual models are there when you want to
be explicit:

| Model | Behaviour |
| --- | --- |
| `switchboard-auto` | Full policy: judge the request, pick the model, switch if it is worth it |
| `switchboard-fast` | Never leave the loaded model |
| `switchboard-deep` | Prefer the strongest reasoning model |
| `switchboard-medical` | Prefer the medical specialist |
| `qwen3.5-9b`, `qwen3.8-27b`, `medgemma-27b` | Exactly that model, loading it if needed |

cptr's client timeouts are worth knowing: 300 s stream read, 30 s connect, and
**15 s** for the small non-streaming call it uses to generate chat titles. That
last one is why Switchboard never makes a short non-streaming request wait for
a model load (`routing.guards.utility_max_tokens`).

### Undo

```powershell
.\scripts\integrate-cptr.ps1 -Remove    # with cptr stopped
```

## C:\AI\start.ps1 (done)

A marker-bracketed block was added before the final health check. It starts
Switchboard after cptr, and it cannot break the stack: it checks the script
exists, wraps everything in try/catch, and only warns on failure.

```powershell
.\scripts\integrate-start.ps1 -Remove   # to take it out
```

Starting Switchboard by hand is still just `scripts\start.ps1`.

## MCP (done)

Switchboard serves MCP over streamable HTTP at `/mcp`, the transport cptr's
client uses. Verified against cptr's own client library (mcp 1.29):

```
server: switchboard | protocol: 2025-11-25
tools: switchboard_route, switchboard_evaluate_step, switchboard_status
```

| Tool | Purpose |
| --- | --- |
| `switchboard_route` | Which local model suits this task, and would a switch be worth it. Never switches. |
| `switchboard_evaluate_step` | Did that step work: continue, retry, retry_different, escalate, ask_user, stop_complete, stop_failed |
| `switchboard_status` | What is loaded, what is healthy, which providers answer |

Routing through MCP is advice only - an MCP tool call never restarts
llama-server. An agent that wants a model loaded asks the HTTP API
(`POST /agent/switch`).

### Adding it to another MCP client

Point it at `http://127.0.0.1:8002/mcp`, transport "streamable HTTP", no auth.

## Open WebUI (not applicable)

Open WebUI is **not installed on this machine** - not on PATH, not in
site-packages, not anywhere under the user profile or Program Files. (cptr is
published by Open WebUI, Inc., which is the likely source of the confusion.)

If it is installed later, it takes an OpenAI-compatible connection:

* Settings → Connections → OpenAI API
* Base URL: `http://127.0.0.1:8002/v1`
* API key: any non-empty string (Switchboard does not check it)
* The model list will include `switchboard-auto` and the real models.

## Open Terminal MCP (not applicable)

No such server exists on this machine. Switchboard adds an MCP server; it
removes none.

## Anything that speaks OpenAI

```bash
curl http://127.0.0.1:8002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"switchboard-auto","messages":[{"role":"user","content":"hello"}],"stream":true}'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8002/v1", api_key="not-checked")
client.chat.completions.create(model="switchboard-auto", messages=[...])
```

Both `/v1/chat/completions` and `/chat/completions` work, because cptr builds
the second form.

Every response carries the decision in its headers:

```
X-Switchboard-Model: qwen3.5-9b
X-Switchboard-Backend: rules
X-Switchboard-Domain: coding
X-Switchboard-Difficulty: 0.85
X-Switchboard-Switched: false
X-Switchboard-Reason: Stayed on qwen3.5-9b: already loaded, and a coding task at
  difficulty 0.8 sits within its 2.0 ceiling. qwen3.8-27b would cost ~20s to
  load for +0.04 expected quality.
```

## The dashboard at :9090

Switchboard depends on the existing dashboard for one thing: performing
switches, through the same `POST /api/models/load` its Models tab uses. With
the dashboard down, Switchboard routes to whatever is already loaded and says
so in `doctor`. It never starts llama-server itself.
