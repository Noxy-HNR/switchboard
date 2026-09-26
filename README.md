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

For normal use on this machine, start the complete stack:

```powershell
C:\AI\start.ps1
```

This checks llama.cpp, starts cptr and Switchboard as needed, and opens the
interfaces. For an unattended startup with no windows or final prompt:

```powershell
C:\AI\start.ps1 -NoUi -NoPause
```

If the rest of the stack is already running, start only Switchboard:

```powershell
cd C:\AI\Switchboard
.\scripts\setup.ps1      # once: creates .venv, checks the machine
.\scripts\start.ps1
```

The current machine's operational settings and validation are in
[OPERATIONS.md](docs/OPERATIONS.md).

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
   each model runs, what a load costs, how much RAM is free. Unknown speeds and
   load costs use explicitly documented estimates until measurements exist.
3. **Policy** - combines both into one choice and one sentence explaining it.

A switch has to clear three bars: it must be worth more than the seconds it
costs, it must actually improve the answer, and the moment has to be right
(server idle, conversation settled, not a short utility call). While
Lecture Notes is recording, Switchboard refuses requests to load a model over
9 GB. This guard applies to Switchboard's switches; the stack dashboard's own
Models tab and a large model already resident remain outside its control.

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
| `qwen3.5-9b` | general, fast | 20-38 tok/s measured | fully GPU resident, the daily driver; reloads in ~11 s |
| `qwen3.8-27b` | deep reasoning | ~10 tok/s | partially offloaded, ~3x slower |
| `medgemma-27b` | medical specialist | ~5 tok/s | its template cannot express tool calls, so tool requests never route here |

Capability priors live in `config/models.toml` and are meant to be edited.
Speeds and load times are measured from real traffic and stored in SQLite; a
seeded estimate is replaced the first time the real thing is observed.

## Settings and keys

Open your dashboard at `http://127.0.0.1:9090` and click the **Settings** tab.
(The same panel is on Switchboard's own page, `http://127.0.0.1:8002/dashboard`,
under **Settings**.) Everything there applies at once, without a restart:

* **API keys** - paste a key, save, then **Test** it. Keys are written to
  `.env` in this folder and never shown again; the page only says whether one
  is set and its last four characters.
* **Settings** in groups: Routing, Models (which models Switchboard may
  switch to on its own), Switch guards (including the lecture-recording
  guard), Policy, Decisions (backends, fast path, cache, timeouts), Jev (cloud),
  Privacy, and Data (how long routing history is kept; older rows are pruned
  daily). The key row also shows the last day's Jev calls and estimated cost.
  They are saved to `config/local.toml`, layered over
  `config/switchboard.toml`; **Reset** next to a setting (or deleting
  `local.toml`) goes back to the file's value.

Both files are git-ignored. A setting or key that the environment sets is
shown as locked, because the environment wins over both files.

## Configuration

`config/switchboard.toml`, commented throughout, holds the defaults and
everything the dashboard does not change (ports, paths, addresses). Values are
read in this order, later ones winning: `switchboard.toml`, `local.toml`,
`.env`, then `SWITCHBOARD_<SECTION>__<KEY>` environment variables.

The two settings most worth knowing:

```toml
[routing]
mode = "active"    # "shadow" = never switch, just record what it would have done

[policy]
value_of_quality_s = 400.0   # how many seconds of waiting a better answer is worth
```

A TypeSafe key (saved in Settings, or `TYPESAFE_API_KEY` in `.env`) enables
the Jev backend. Without it, everything stays on this machine.

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
