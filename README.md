# Switchboard

A routing and supervisory layer in front of the local llama.cpp stack in
`C:\AI`. It decides which local model should answer a request, whether a model
switch is worth its measured cost, and whether an agent step made progress.

It speaks the OpenAI API, so anything that already talks to `llama-server` can
talk to Switchboard instead and get routing for free.

```
cptr / scripts / agents          ->  Switchboard :8002  ->  llama-server :10000
                                          |                   (one model resident)
                                     decisions + policy
                                          |
                                     dashboard :9090 (performs switches)
```

## Start it

```powershell
cd C:\AI\Switchboard
.\scripts\setup.ps1      # once: creates .venv, checks the machine
.\scripts\start.ps1
```

That is the normal way. `C:\AI\start.ps1` also starts it now, after cptr.

| | |
| --- | --- |
| OpenAI endpoint | `http://127.0.0.1:8002/v1` (model: `switchboard-auto`) |
| Dashboard | `http://127.0.0.1:8002/dashboard` |
| API docs | `http://127.0.0.1:8002/docs` |
| MCP | `http://127.0.0.1:8002/mcp` |

```powershell
.\scripts\stop.ps1       # stops only Switchboard
.\scripts\doctor.ps1     # diagnoses the whole stack, works with it all stopped
```

## Use it

```bash
curl http://127.0.0.1:8002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"switchboard-auto","messages":[{"role":"user","content":"hello"}]}'
```

In cptr: pick the **Switchboard** connection and the model `switchboard-auto`.

Every answer carries its reasoning in the response headers:

```
X-Switchboard-Model: qwen3.5-9b
X-Switchboard-Reason: Stayed on qwen3.5-9b: already loaded, and a coding task at
  difficulty 0.8 sits within its 2.0 ceiling. qwen3.8-27b would cost ~20s to
  load for +0.04 expected quality.
```

## How it decides

Three things are kept apart, because conflating them is what makes routers
mysterious:

1. **Semantic judgments** - what kind of work is this, how hard, does it need
   tools. Answered by Jev (cloud, if a key is set), then TypeSafe's System One
   Adapter running on the model already loaded, then a rule engine
   that cannot fail.
2. **Deterministic facts** - what is loaded, what fits in context, how fast
   each model runs, what a load costs, how much RAM is free. Measured, never
   guessed.
3. **Policy** - combines both into one choice and one sentence explaining it.

A switch has to clear three bars: it must be worth more than the seconds it
costs, it must actually improve the answer, and the moment has to be right
(server idle, conversation settled, not a short utility call).

Most requests never pay for a semantic decision at all - the rule engine
settles them in about a millisecond, and only genuinely hard or medical
requests consult a slower backend.

## Command line

```powershell
uv run switchboard status        # what is loaded, what is healthy
uv run switchboard models        # the registry, with measured speeds
uv run switchboard providers     # decision backends, breaker state, cost
uv run switchboard route "..."   # what would this be routed to, and why
uv run switchboard step --goal ... --action ... --result ...
uv run switchboard doctor        # diagnose everything
uv run switchboard eval          # measure routing against fixtures
uv run switchboard benchmark     # measure model speed (--switches for loads)
uv run switchboard recent        # recent routing decisions
```

## The models it routes to

Read from `C:\AI\Config\models.json` and the GGUF files themselves, never
hardcoded:

| Model | Role | Speed | Notes |
| --- | --- | --- | --- |
| `qwen3.5-9b` | general, fast | ~39 tok/s | fully GPU resident, the daily driver |
| `qwen3.8-27b` | deep reasoning | ~10 tok/s | partially offloaded, ~3x slower |
| `medgemma-27b` | medical specialist | ~5 tok/s | its template cannot express tool calls, so tool requests never route here |

Capability priors live in `config/models.toml` and are meant to be edited.
Speeds and load times are measured from real traffic and stored in SQLite; a
seeded estimate is replaced the first time the real thing is observed.

## Configuration

`config/switchboard.toml`, commented throughout. Anything there can be
overridden by `SWITCHBOARD_<SECTION>__<KEY>` in the environment or `.env`.

The two settings most worth knowing:

```toml
[routing]
mode = "active"    # "shadow" = never switch, just record what it would have done

[policy]
value_of_quality_s = 400.0   # how many seconds of waiting a better answer is worth
```

`TYPESAFE_API_KEY` in `.env` enables the Jev backend. Without it, everything
stays on this machine.

## Documentation

| | |
| --- | --- |
| [DISCOVERY.md](docs/DISCOVERY.md) | what this machine looked like before any of this |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | how it works, and what was deliberately not built |
| [DEPENDENCY-VERIFICATION.md](docs/DEPENDENCY-VERIFICATION.md) | every external API, checked against current docs |
| [INTEGRATION.md](docs/INTEGRATION.md) | cptr, MCP, start.ps1, and anything OpenAI-compatible |
| [PRIVACY.md](docs/PRIVACY.md) | what leaves the machine and what is stored |
| [ROLLBACK.md](docs/ROLLBACK.md) | undoing every change, file by file |
| [EXPERIMENT-LOGITS.md](docs/EXPERIMENT-LOGITS.md) | the first-token-probability backend: measured, and what it does not claim |
| [ACCEPTANCE.md](docs/ACCEPTANCE.md) | every requirement, with the evidence |

## Tests

```powershell
.\scripts\test.ps1
```

Offline by default: every test uses mock transports, so the suite passes with
the whole stack stopped and no network.
