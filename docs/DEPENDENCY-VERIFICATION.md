# Dependency verification

Every external interface Switchboard codes against, checked against the current
official source on **2026-09-18 / 2026-09-19** (the TypeSafe SDK and System One
Adapter sections re-checked on **2026-09-22** for typesafe-sdk 0.7.1 and
system-one-adapter 0.2.1) rather than assumed. Where the
original brief's expectation turned out to be right, that is recorded too —
confirming an assumption is a result.

## TypeSafe / Jev

Checked against <https://docs.typesafe.ai> (`llms.txt` index, then the API,
Models, primitives, confidence and jaggedness pages) and against the installed
SDK in `.venv/Lib/site-packages/typesafe_sdk`.

| Expectation from the brief | Verified state |
| --- | --- |
| `Jev`, `System One` | Correct. Jev is TypeSafe's System One model; System One models return typed answers with calibrated probabilities instead of text. |
| `typesafe-sdk`, Python, async client | Correct. `pip install typesafe-sdk`; `AsyncTypeSafeClient` and `TypeSafeClient`. |
| `Noul`, `Choice`, `Score` | Correct, and they are the only three question types. |
| `TYPESAFE_API_KEY` | Correct (`typesafe_sdk.constants.API_KEY_ENV`). Also `TYPESAFE_BASE_URL`, `TYPESAFE_DEFAULT_MODEL`, `TYPESAFE_LOG_LEVEL`. |
| `jev-latest` | Correct; it is the SDK default and currently resolves to `jev-1.13.0`. `jev-preview` points at the same build. |
| An official System One Adapter | **Exists**: `typesafe-ai/system-one-adapter-python`, v0.2.1, MIT. |

### API contract used

`POST https://api.typesafe.ai/v1/systemone`, `Authorization: Bearer <key>`:

```json
{ "state": <string | object | array>, "model": "jev-latest",
  "questions": { "<id>": { "type": "noul|choice|score", "instructions": ..., "criteria": ... } } }
```

- **Noul** answer: `{"type":"noul","noul":0.92}` — a probability, and **no
  confidence field**. Code that reads `.confidence` on every answer breaks here;
  Switchboard's `TaskDecision` keeps Noul values in plain float fields.
- **Choice** answer: `choice`, `probabilities` (sums to 1), `confidence`.
- **Score** answer: `score` (probability-weighted, can land between levels),
  `legend`, `probabilities`, `confidence`. Switchboard uses `score` directly and
  never interpolates a magnitude from it — the docs warn Score is weak at
  numeric calibration.
- Errors: 401, 422, 429, 529. The SDK maps these to
  `TypeSafeAuthenticationError`, `TypeSafeUnprocessableEntityError`,
  `TypeSafeRateLimitError` (with `retry_after_ms`), `TypeSafeInternalServerError`.
- Usage: `{"input_tokens", "output_tokens"}`.

### Model facts that drive configuration

| Property | Value | Where it shows up in Switchboard |
| --- | --- | --- |
| Price | **$0.042 per Mtok input**, output free | `decisions.jev.usd_per_input_mtok`; cost per decision is ~1e-5 USD |
| Rate limits | 250k tok/s, 1200 req/min, "adjusting dynamically" | breaker + retry policy |
| Context | 64k per request; 32k for state plus longest question | `privacy.max_state_chars` (1500) is far inside this |
| Input | **Text only** | images in a request are recorded as a flag, never forwarded |

### SDK version pinned, deliberately

`typesafe-sdk==0.7.1`. The SDK is days old and has already shipped two breaking
changes (0.6.0 changed `Score.criteria` to an ordered sequence; **0.7.0 moved
serialisation from `msgspec` to `pydantic`** on 2026-09-18). 0.7.1
(2026-09-21) is a security release with no request, response or type changes:
the API key is stripped and validated when the client is created and masked in
connection-error messages and exception chains, which is what lets the Jev
backend include a plain `TypeSafeError`'s own message when describing a
construction failure. Its debug logging writes request bodies unredacted, so
the Jev backend holds the `typesafe_sdk` logger at INFO or above whatever
`TYPESAFE_LOG_LEVEL` says. It depends on
`httpx2`, not `httpx`, and `AsyncTypeSafeClient(transport=...)` accepts an
`httpx2.AsyncBaseTransport` — which is what the Jev backend's tests drive with
`httpx2.MockTransport`.

### Guidance that changed the design

From the API docs, the patterns pages and `model-jaggedness/jev-1.13`:

- **Batch every question for one judgment into one call.** Jev ingests the state
  once and answers all questions in parallel; the docs' own cookbook measures
  batching as 12x cheaper and 10x faster. Switchboard makes exactly one call per
  task decision (4 questions) and one per step decision (4 questions).
- **Keep arithmetic in code.** Jev "is not a calculator" and "reads dates as
  text". Context sizes, switch costs, VRAM and RAM are therefore computed in
  Python and never asked about.
- **Small state beats large state, and only fields a question names.**
  Accuracy falls as irrelevant content grows. Switchboard sends the latest user
  turn and, when there is one, the previous one, plus `has_images` when true -
  exactly the paths the questions quote (`decisions/state.py`). Tool names and
  the turn number used to be sent too; no question judged them, so they went.
  Pasted machine output inside a turn is replaced by a one-line
  `[pasted ... omitted: N lines]` marker the questions explain.
- **Do not expect structural invariants.** A Noul and a "yes/no" Choice on the
  same question return different numbers, so a threshold tuned on one must not
  be carried to the other. Switchboard keeps each question's answer in its own
  field and never compares across primitives.
- **Write the exact condition.** Jev answers the words as written, so every
  question's `instructions` and `criteria` are explicit about boundary cases
  (see `decisions/schema.py`, where all the question text lives).

## System One Adapter (the local fallback)

`system-one-adapter==0.2.1`, official, MIT, depends on `typesafe-sdk>=0.7.0`.
0.2.1 rejects incomplete completions, stops retrying refusals, and makes
`ProviderResult` token counts `int | None`; the local provider mirrors all
three (a `finish_reason: "length"` reply is refused as invalid, and a count
llama-server did not report stays `None` instead of becoming 0).

- It is a drop-in `system_one` implementation backed by ordinary LLMs, which is
  exactly the "local fallback" role: the same questions, the same answer types,
  a different engine.
- `SystemOneAdapterClient(structured_outputs=..., llm_answer_mode="probabilities"|"discrete", normalize_probabilities=...)`.
- **Switchboard runs it in prompted mode** (`structured_outputs=False`). In
  structured mode the questions and criteria exist only as `description`s in
  the JSON schema, and llama.cpp turns `response_format` into a sampling
  grammar without ever showing the schema to the model, so the model would see
  no questions at all. Prompted mode appends the adapter's own schema template
  to its system message; the provider still attaches the grammar, so the reply
  always parses. The adapter's trace then records exactly what the model saw.
- Its `openai`/`anthropic` extras are **not installed**. The provider protocol
  (`providers/base.py`) needs only `model_name`, `async request(messages, *, schema, structured)`
  and `translate_error`, so Switchboard ships its own `LlamaCppProvider` over
  `httpx` — one HTTP stack, no OpenAI SDK, and full control of the request body.
- **Its probabilities are not calibrated.** In `"probabilities"` mode the LLM
  writes numbers into JSON; they are verbalised, not measured. Switchboard
  defaults to `"discrete"` and, in that mode, leaves probability dicts empty and
  confidence `None` rather than inventing values.

## llama.cpp server (build 11041, commit `4fea119de`)

Verified from `--help` on the installed binary and from `tools/server/README.md`
in the matching source checkout.

| Capability | State | Use |
| --- | --- | --- |
| `POST /v1/chat/completions` and `/chat/completions` | both present | the proxy serves both prefixes for the same reason |
| `chat_template_kwargs: {"enable_thinking": false}` | supported | disables the Qwen3.5 thinking pass for decision calls |
| `reasoning_effort: "none"` | supported (sets `enable_thinking=false`) | sent alongside, so either build behaves |
| `logprobs: true` + `top_logprobs: N` | supported; maps to llama.cpp `n_probs` | the experimental logit backend |
| `response_format: {"type":"json_schema", ...}` | supported | grammar-constrained structured output for the local backend |
| `timings` in responses (`predicted_per_second`) | present | real tokens/sec measurements from ordinary traffic |
| `/slots` with `is_processing` | enabled by default | "is the server busy?" — a switch guard |
| `/props` | present | resident model path and context size |
| `--cache-ram` (default 8192 MiB) | present | why a decision call does not destroy the chat's KV cache |
| **Router mode** (`--models-dir`, `--models-preset`, `--models-max`, `POST /models/load`, autoload) | **present in this build** | **deliberately not used** — see below |

### Why router mode is not used

This build genuinely supports running `llama-server` with no model, serving
several models and loading them on demand. It was rejected for now, not
overlooked:

- Using it means changing how `llama-server` is launched, which is the one thing
  the brief asks to preserve and the thing cptr depends on. A proxy in front of
  the existing configuration achieves the same routing with no change to the
  stack.
- With 12 GB of VRAM, `--models-max` would have to be 1, so the router would be
  doing the same "unload one, load the other" work the dashboard already does —
  with a second code path for starting llama-server, which `Build-LlamaArgs.ps1`
  exists specifically to avoid.
- Per-model tuning (context size, GPU layers, KV cache type) currently lives in
  `Config\models.json` and is edited from the dashboard. Router presets are a
  second, parallel place to configure the same things.

Switchboard *detects* router mode (`upstream.UpstreamStatus.router_mode`) and
reports it, so the day the stack moves to it, the routing layer already knows.
Making it the switch executor is a small addition, described in
[ARCHITECTURE.md](ARCHITECTURE.md#if-the-stack-moves-to-router-mode).

## MCP

- `mcp` **1.30.0**, pinned `>=1.29,<2`. Version 2.x exists and renames `FastMCP`
  to `MCPServer`, but cptr — the only MCP client on this machine — ships
  `mcp 1.29`, so staying on the same major version keeps the protocol
  negotiation path well-trodden. Revisit when cptr moves.
- Transport: **streamable HTTP**, which is what cptr's client
  (`mcp.client.streamable_http.streamablehttp_client`) uses, mounted at `/mcp`.
- A mounted sub-application does not receive its own lifespan, so the session
  manager is run from the parent app's lifespan (`app.py`).

## Everything else

| Package | Version | Why |
| --- | --- | --- |
| fastapi | 0.141.1 | the API and proxy; OpenAPI docs come free |
| httpx | 0.28.1 | one async HTTP stack; `MockTransport` makes every network test offline |
| uvicorn | 0.53.0 | the server |
| typer | 0.27.2 | the CLI |
| pydantic | 2.13.5 | config and API schemas (already a FastAPI dependency) |
| pytest / pytest-asyncio | 9.1.1 / 1.4.0 | tests, `asyncio_mode = auto` |

No database driver: SQLite is in the standard library. No psutil: memory comes
from `GlobalMemoryStatusEx` via `ctypes`, and GPU facts from `nvidia-smi`.
