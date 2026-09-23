# Architecture

## The shape of it

```
cptr / scripts / agents / CLI
        |
        v
  Switchboard  :8002
        |
   +----+--------------------------------+
   |                                     |
 decision layer                    routing policy
 (semantic judgments)              (judgments + facts)
   |                                     |
 jev -> local -> rules            registry, upstream, host
   |                                     |
   +----------------+--------------------+
                    v
            switch executor  --->  dashboard :9090  ---> restarts
                    |                                    llama-server
                    v
            llama-server :10000  (one model resident)
```

Switchboard is a proxy and an advisor. It never starts `llama-server` itself:
when a switch is warranted it asks the dashboard, which is the same path the
Models tab and `model-manage.ps1` use. That keeps exactly one piece of code on
this machine responsible for launching llama-server.

## The one idea

Three things are kept apart, because conflating them is what makes routers
mysterious:

| | What it is | Where it lives | Example |
| --- | --- | --- | --- |
| **Semantic judgment** | something only a model can read out of language | `decisions/` | "this is a medical question at difficulty 2.4" |
| **Deterministic fact** | something code knows exactly | `registry.py`, `upstream.py`, `system.py` | "medgemma is not loaded; loading it took 27.6 s last time; it needs ~7.7 GB of host RAM" |
| **Policy** | what to do given both | `policy.py` | "stay on the 9B: the gain does not cover the load" |

A decision backend is never asked how much VRAM is free, and the policy never
guesses what a request is about.

## Modules

| Module | Responsibility |
| --- | --- |
| `config.py` | `switchboard.toml`, then the dashboard's `local.toml`, then `.env`, then the environment; one `Config` object, no globals |
| `settings.py` | the dashboard's Settings API: which settings it may change, `local.toml` and `.env` storage, key handling, live re-application through `Router.reconfigure` |
| `privacy.py` | redaction, hashing, previews, cloud-state preparation |
| `gguf.py` | GGUF header reader: native context, layer count, chat template |
| `registry.py` | the model registry, built from `Config\models.json` + GGUF + `config/models.toml` + measurements |
| `system.py` | host facts: free RAM (`GlobalMemoryStatusEx`), GPU (`nvidia-smi`), cached |
| `upstream.py` | the routing and proxy path's llama-server client: health, resident model, busy, router detection, forwarded completions (the `local` and `logit` decision backends post their own `/v1/chat/completions` requests) |
| `executor.py` | how a switch is carried out (`DashboardExecutor`, `NullExecutor`) |
| `decisions/schema.py` | **the questions**, shared verbatim by every backend, and the answer types |
| `decisions/base.py` | the backend contract: availability, `BackendError`, contexts |
| `decisions/state.py` | the one state every backend sends: the fields the questions name, with fixed budget shares |
| `decisions/jev.py` | TypeSafe Jev (cloud) |
| `decisions/local.py` | System One Adapter over the resident llama.cpp model |
| `decisions/logit.py` | experimental: first-token probabilities instead of generated JSON |
| `decisions/rules.py` | deterministic heuristics; cannot fail |
| `decisions/chain.py` | ordered fallback, budget, circuit breaker, cache |
| `policy.py` | scoring, feasibility, guards, the sentence explaining the choice |
| `router.py` | the service: decide, switch under a lock, record, supervise steps |
| `proxy.py` | the OpenAI-compatible surface, streaming and all |
| `api.py` | the supervisory API for agents |
| `mcp_server.py` | the same supervision as MCP tools |
| `store.py` | SQLite: routes, attempts, switches, step evals, measurements, sessions |
| `diagnostics.py` | `doctor` |
| `evals/` | fixtures, the eval runner, the benchmark tool |

## The decision surface

Four questions per request, asked in **one** call by Jev and by the local
adapter (the TypeSafe docs measure batching as an order of magnitude cheaper
and faster). The experimental logit backend asks them one per request:

- `domain` — Choice over general / coding / medical / writing / research / system_ops
- `difficulty` — Score over four described levels, 0 (trivial) to 3 (hard)
- `tool_need` — Noul: does answering require acting outside the conversation?
- `ambiguity` — Noul: would a competent assistant have to ask first?

Four more for a finished agent step: `outcome` (Choice), `progress`,
`complete`, `needs_new_approach` (Nouls). The *recommendation* — continue,
retry, retry_different, escalate, ask_user, stop_complete, stop_failed — is
computed in code from those judgments plus hard facts, because "exit code 1" is
not an opinion. A non-zero exit code overrides a model that claims success, and
the response says so (`facts_overrode_model`).

The question text lives in `decisions/schema.py` and is shared verbatim by all
backends, and so is the **state** (`decisions/state.py`): every backend sends
the same fields, under the same names the questions quote. Identical questions
over differently named state would not be comparable answers.

The state holds only what a question names. `request.message`, plus
`request.previous_message` and `request.has_images` when present; for a step,
`task.goal`, `step.action`, `step.result`, `step.exit_code` and
`earlier_attempts`. The system prompt, the client's tool roster, the turn
number and the retry count are not evidence for any question, so no model sees
them. The policy reads only whether tools are present (`has_tools`) and a
prompt-size estimate that includes the system prompt's length; the step
recommendation reads the retry count; the turn number is not used. A shared
character budget is split in fixed shares: the previous message is capped at
300 characters and the current message gets the rest; the step result gets
whatever goal, action and earlier attempts (each capped) leave. An optional
field (the previous message, an earlier attempt) that cleans to nothing is
dropped rather than sent as an empty string a literal reader might take as
evidence; the message, goal, action and result are always sent, and an empty
`step.result` is kept on purpose because an empty result is itself evidence.
How the text is
cleaned is the backend's business: Jev minimizes and redacts
([PRIVACY.md](PRIVACY.md)), the local backends only truncate.

Every decision says whether its probabilities are `calibrated`. Only Jev's are
(it is trained for that); the adapter's discrete mode has none, and the logit
backend's are a next-token distribution. The policy weighs a distribution only
when it is calibrated and otherwise uses the point answer.

## The backend chain

`jev -> local -> rules`, first real answer wins. The logit backend is
implemented but off ([EXPERIMENT-LOGITS.md](EXPERIMENT-LOGITS.md)).

- **Fast path first.** The rules engine answers in under a millisecond; when it
  is confident, the request is easy, and no must-confirm domain (medical) holds
  even a 20% share, its answer stands and no model is asked.
- **Budget, not just timeouts.** The whole semantic layer gets
  `decisions.budget_ms` (6 s). Jev's own retry sequence is capped at half of it,
  so a failing provider still leaves time for the local backend. Each backend gets the smaller of its own timeout
  and what is left. The floor (`rules`) runs even when the budget is spent,
  because a decision must exist.
- **Breaker.** Repeated failures park a backend for a cooldown. A *refusal*
  ("no API key", "resident model too slow") is not a failure and never opens it.
- **Cache.** Identical question, text and resident model within the TTL reuses
  the answer. Keys are SHA-256 digests (no text is kept as a key) and entries
  are copied in and out, so a caller cannot mutate a cached decision.
- **Floor.** `rules` cannot fail and is forced to the end of the chain even if
  the configuration forgets it.

The local backend runs the System One Adapter in **prompted** mode
(`structured_outputs=False`) with the answer schema always attached as a
sampling grammar. llama.cpp uses `response_format` only as a grammar and never
shows the schema to the model, so in the adapter's native mode the questions,
which live in the schema's descriptions, would never reach the model. Prompted
mode has the adapter put them in its single system message, and its own trace
records exactly what the model saw. A reply cut off at `max_tokens` is refused
as invalid and not retried; like every invalid answer it counts toward the
breaker, so three in a row park the backend for `breaker_cooldown_s` rather
than spend each request's budget on it. Token counts llama-server did not
report are left out rather than recorded as zero.

When is the local backend allowed to run? Only when llama-server is healthy,
**idle** (`--parallel 1` means a decision call would queue behind a real
generation) and the resident model is fast enough to be worth asking
(`min_resident_gen_tps`, default 15 tok/s — a 5 tok/s MedGemma would spend more
time deciding than it saves).

## The policy

For each model that is feasible for this request:

```
utility(m) = value_of_quality_s * E[P(good answer | m)] - seconds_of_waiting(m)
E[P(good answer | m)] = sum over domain d, level l of w(d) * w(l) * q(m, d, l)
q(m, d, l) = sigmoid(capability_slope * (ceiling(m, d) - l))
seconds_of_waiting(m) = switch_cost(m) + prompt_time(m) + g * generation_time(m)
```

where `g` is `generation_wait_weight` (default 0.35) for a streaming request,
whose answer starts arriving at once, and 1.0 for a non-streaming one, whose
caller waits for all of it.

The weights `w` are the calibrated distributions when the backend has them
(Jev): a request that is 55% general and 45% medical is scored as exactly that,
and each difficulty level is scored at its own level before weighting, because
a Score does not license interpolating between levels. Without a calibrated
distribution the point answer gets all the weight. When a calibrated domain
split mattered (a second domain at 15% or more), the stay or switch reason
names it ("general 55%, medical 45%"); uncalibrated decisions and guard-blocked
reasons name no split.

Quality is expressed **in seconds** so it can be compared with waiting, and
`value_of_quality_s` (default 400) is the single knob that says how much a better
answer is worth. `ceiling(m, domain)` is a prior from `config/models.toml` — an
editable policy parameter, clearly not a benchmark. Everything else is measured
where a measurement exists: `switch_cost` from observed loads (file size ×
`default_load_s_per_gb` until a model has been loaded), `generation_time` from
observed tokens/sec (5 tok/s assumed when unknown), and `prompt_time` from
prompt tokens/sec (seeded estimates in `config/models.toml` until measured).

**Feasibility is checked before scoring**, and infeasibility is a hard no:

- context: estimated prompt + expected answer must fit the model's configured
  context (a 25k-token conversation simply cannot use the 27B's 20480);
- tools: a request carrying tool definitions never goes to a model whose chat
  template cannot express them (MedGemma);
- host RAM: a model whose estimated CPU-resident share does not fit gets a
  warning (default) or is excluded (`ram_guard = "block"`).

**Then three guards stand between a preference and a restart:**

1. `hysteresis_seconds` — utility must beat staying put by a margin;
2. `min_quality_gain` — and the answer must actually get better;
3. `min_dwell_seconds`, `never_switch_when_busy`, `utility_max_tokens` — and the
   moment has to be right: not within `min_dwell_seconds` of this
   conversation's own last switch (dwell is tracked per session, so another
   conversation, or a request without one, is not held), not while
   llama-server is generating for someone else, and never for a short
   non-streaming utility call while a usable model is resident (cptr's
   chat-title requests time out at 15 s). When the resident model cannot serve
   the request at all, or nothing is loaded, only the busy guard applies.

When a guard blocks a switch, that is recorded, not discarded. The guard's
sentence is stored in `routes.reason`, the model it would have loaded is sent
as the `X-Switchboard-Would-Switch-To` response header, and `POST /agent/route`
returns both `would_switch_to` and `blocked_by` in its `routing` object. That
is what makes shadow mode useful.

**Stickiness** comes from the conversation's running difficulty: the policy uses
`max(instant difficulty, session EWMA)`, so one short follow-up in a hard
conversation does not drag it back to the small model and then immediately back
up again.

## Streaming

Upstream bytes are relayed as they arrive; nothing is buffered. Token counts are
recovered by watching the bytes go past (`UsageWatcher`) — real `usage` when the
client asked for it, otherwise counted content deltas — because rewriting the
client's request to add `stream_options` would change what the client receives.

While a model loads, a streaming request gets SSE **comment** lines
(`: switchboard preparing a model (10s)`), which every compliant parser ignores,
including cptr's. A 25-second load looks like a slow start instead of a dead
socket.

## Deliberate deviations from the brief

| Brief | Built | Why |
| --- | --- | --- |
| "Open WebUI / Open Terminal MCP integration" | documented as NOT_APPLICABLE | Neither is installed on this machine ([DISCOVERY.md](DISCOVERY.md)) |
| "`C:\AI\start-ai.ps1`" | `C:\AI\start.ps1` | that is the real filename |
| Switchboard restarts llama-server | it asks the dashboard to | one code path for launching llama-server, and `Build-LlamaArgs.ps1` stays the single source of truth for arguments |
| llama.cpp router mode as the multi-model mechanism | detected, reported, not used | it would change the launch path the brief asks to preserve; see [DEPENDENCY-VERIFICATION.md](DEPENDENCY-VERIFICATION.md#why-router-mode-is-not-used) |
| Local fallback = System One Adapter | that, plus an availability gate | on this machine the adapter's cost depends entirely on which model is resident, so it refuses when the resident model is slow or busy |
| Experimental logit backend | implemented, **off** | its probabilities are a next-token distribution that could not be validated as calibrated, and it measured below the adapter backend ([EXPERIMENT-LOGITS.md](EXPERIMENT-LOGITS.md)) |
| Shadow mode as the default | `mode = "active"` | you chose auto-switching through the dashboard; shadow remains one setting away. Requests that name a model, and all traffic in `mode = "off"`, bypass the decision layer (`X-Switchboard-Backend: bypass`) and are not evaluated |

## If the stack moves to router mode

`upstream.py` already detects it (`/v1/models` entries carrying a `status`
object). To use it, add a `RouterExecutor` to `executor.py` that posts to
llama-server's own `/models/load` and returns the same `SwitchResult`; the
policy, registry and proxy need no changes. `build_executor()` would pick it
when `UpstreamStatus.router_mode` is true.
