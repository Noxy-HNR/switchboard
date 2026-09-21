# Discovery: this machine before Switchboard

Everything below was read from the machine on **2026-09-18 / 2026-09-19**, before
Switchboard changed anything. It is the "what did it look like before" record.
Where a number is quoted, the command or file it came from is named.

## Machine

| Fact | Value |
| --- | --- |
| Hostname | `MSI` |
| Model | MSI Vector 16 HX AI A2XWHG |
| OS | Windows 11 Home 10.0.26200 (build 26200) |
| CPU | Intel Core Ultra 7 255HX, 20 cores / 20 logical |
| RAM | **15.4 GB total** (4.3 GB free with the stack stopped) |
| GPU 0 | NVIDIA GeForce RTX 5070 Ti Laptop, **12227 MiB VRAM**, driver 616.92, CUDA UMD 13.4 |
| GPU 1 | Intel Graphics (iGPU), driver 32.0.101.6629 |
| Python | 3.13.15 (`C:\Users\braxt\AppData\Local\Programs\Python\Python313`), 3.12 via the Store |
| uv | 0.12.3 |
| git | 2.55.0 |
| Node / npm | **not installed** |

`nvidia-smi` at rest: `P0, 17.5 W / 140 W, 0 MiB used, 0% util`. The GPU running
at low power is [a deliberate choice by the user](#notes-carried-in-from-earlier-sessions),
not a fault, and Switchboard never comments on it.

## The AI stack (`C:\AI`)

A git repository (`github.com/Noxy-HNR/ai-workspace`) holding scripts, the
dashboard and config; models, logs and nested projects are gitignored.

| Component | Where | Notes |
| --- | --- | --- |
| llama.cpp | `C:\AI\Tools\llama-native\bin\llama-server.exe` | **version 0.4.1-dev, build 11041, commit `4fea119de`**, MSVC x64, CUDA |
| llama.cpp source | `C:\AI\Tools\llama-src` | the same commit, so its README documents this exact build |
| Launcher | `C:\AI\start.ps1` | starts llama-server, the dashboard, then cptr. **There is no `start-ai.ps1`** |
| Stopper | `C:\AI\stop.ps1` | kills port 10000, the dashboard and cptr |
| Dashboard | `C:\AI\dashboard-server.ps1` -> `:9090` | PowerShell `HttpListener`; owns model loading |
| Argument builder | `C:\AI\Tools\scripts\Build-LlamaArgs.ps1` | dot-sourced by both `start.ps1` and the dashboard so they cannot drift |
| Model config | `C:\AI\Config\models.json` | per-model llama-server parameters plus `activeModel` |
| cptr | pip package `cptr` 0.9.21 -> `:8000` | by Open WebUI, Inc.; config at `~/.cptr/config.toml`, data in `~/.cptr/app.db` |
| Shared NPU/iGPU service | `C:\AI\Tools\npu-services` -> `:8788` | embeddings and person detection for *other* projects; Switchboard does not use it |
| Debug proxy | `C:\AI\llama-proxy.py` | a manual request logger on `:10001`, not running |

**Nothing in the stack was running** when discovery began: no process on 10000,
9090 or 8000, and 0 MiB of VRAM in use.

### Not installed, despite being named in the original brief

- **Open WebUI** — not on PATH, not in site-packages, no directory anywhere under
  the user profile or Program Files. (cptr is *published by* Open WebUI, Inc.,
  which is the likely source of the confusion.)
- **Open Terminal MCP** — no such server anywhere on the machine.
- **Claude Code MCP servers** — `~/.claude.json` defines none; the only MCP
  pieces are cptr's tool server and a Claude Desktop extension (Windows-MCP).

These are recorded as NOT_APPLICABLE in [ACCEPTANCE.md](ACCEPTANCE.md).

### llama-server launch parameters

`start.ps1` and the dashboard both build the command line with
`Build-LlamaArgs.ps1`, from the active model's entry in `Config\models.json`:

```
--model <path> --ctx-size <n> --n-gpu-layers <n> --host 0.0.0.0 --port 10000
--parallel <n> --batch-size <n> --ubatch-size <n> --threads <n>
--cache-type-k <t> --cache-type-v <t> --flash-attn on|off [--jinja]
[--rope-freq-base] [--rope-freq-scale] [extraArgs]
```

Two things follow from this that shape Switchboard's design:

- **`--parallel 1`** for every model: a second request queues behind the first.
  A local decision call therefore competes with real generation, which is why
  the local backend refuses while the server is busy.
- **`--host 0.0.0.0`, no API key**: llama-server logs the warning itself. Not
  changed here; noted in [PRIVACY.md](PRIVACY.md).

### Models actually present

From `C:\AI\models\*.gguf`, their entries in `Config\models.json`, and their
GGUF headers (read by `switchboard.gguf`):

| File | Size | Arch | Native ctx | Configured ctx | GPU layers | Tools in template | Role given |
| --- | ---: | --- | ---: | ---: | --- | --- | --- |
| `Qwen3.5-9B-UD-Q6_K_XL.gguf` | 8.2 GB | qwen35 | 262144 | 114688 | 32/32 (all) | yes | general-fast |
| `Qwen3.8-27B-Uncensored-IQ3_M.gguf` | 11.9 GB | qwen35 | 262144 | 20480 | 60/65 | yes | deep-reasoning |
| `medgemma-27b-it-Q4_K_M.gguf` | 15.4 GB | gemma3 | 131072 | 32768 | 31/62 | **no** | medical-specialist |

`activeModel` at discovery: `medgemma-27b-it-Q4_K_M.gguf`.

`Config\models.json` also contains a **stale entry** for
`Qwen3.5-9B-Q4_K_M.gguf`, whose file no longer exists. Switchboard reports it as
a registry problem and never routes to it; the file was left untouched.

MedGemma's chat template genuinely cannot express tool calls (Gemma-3 lineage) —
this is read from the GGUF, not assumed — so requests carrying tool definitions
are never routed to it.

### Measured performance already on record

These are this machine's own numbers, not vendor claims:

| Model | Generation | Source |
| --- | --- | --- |
| Qwen3.5-9B-UD-Q6_K_XL | 30.2 tok/s | `Logs\overnight-tuning-log.md`, session 5 (2026-09-10), Brave open |
| Qwen3.8-27B-Uncensored | 9.6–10.8 tok/s | same log, session 4 |
| medgemma-27b-it | 4.6–5.6 tok/s | `Logs\llama.stderr.log`, 2026-09-18 |
| medgemma-27b-it **load time** | **27.6 s** (process start to "model loaded") | `Logs\llama.stderr.log`, 2026-09-18 |
| medgemma-27b-it prompt processing | 22.8 tok/s on a 124-token prompt | same |

The tuning log also records two findings that Switchboard encodes as policy:

1. **RAM exhaustion is catastrophic, not gradual.** A 27B partial offload fell
   from ~9.6 tok/s to **1.25 tok/s** when free RAM hit 0.7 GB. Switchboard
   estimates each model's host-RAM share from `gpuLayers / block_count` and
   warns (or blocks, if configured) before switching into that state.
2. **medgemma's file (15.4 GB) exceeds total RAM (15.4 GB).** It works, but it
   is the one model that can put the machine under memory pressure.

### Model switching, as the stack already does it

There is exactly one mechanism, and Switchboard reuses it rather than adding a
second: `POST http://127.0.0.1:9090/api/models/load {"filename": "<file>"}`.
The dashboard's `Apply-ModelSettings` then:

1. writes `activeModel` into `Config\models.json`;
2. stops whatever is on port 10000;
3. starts `llama-server.exe` with that model's arguments, at `High` priority;
4. returns immediately — the load itself takes 15–60 s.

`model-manage.ps1 load <name>` calls the same endpoint. **One model is resident
at a time**; nothing on this machine hot-swaps.

### cptr configuration (`~/.cptr/config.toml`)

- One chat connection: **"Llama.cpp"**, `base_url = http://127.0.0.1:10000`,
  `provider = openai`, `provider_type = llama.cpp`, enabled.
  cptr builds `{base_url}/chat/completions` — **without `/v1`** — which is why
  Switchboard serves both path shapes.
- One MCP tool server: **"Personal Health"**, `type = mcp`,
  `url = http://127.0.0.1:8765/mcp` (streamable HTTP). Untouched.
- Auth is a JWT cookie; the admin API needs a browser session, so it cannot be
  driven from a script. cptr's own `db.py` says the TOML file is authoritative
  and "allows hand-editing", which is the integration path used.
- cptr's client timeouts: stream connect 30 s, **stream read 300 s**, and a
  **15 s** non-streaming timeout for chat-title generation (`max_tokens=100`).
  That 15 s timeout is why Switchboard never makes a short non-streaming call
  wait for a model load.
- Its SSE parser skips any line not starting with `data:`, so the keepalive
  comments Switchboard emits during a load are safe.

> **Note:** `config.toml` stores a Browser Use API key in plaintext. Switchboard
> does not read, copy or transmit it, and its redaction rules would strip a
> `bu_…` token from anything leaving the machine. Flagged for the user only.

### Ports

Listening at discovery: 135, 445, 1462, 2869, 5040, 5357, 7250 (CastSrv), 9080
(Nahimic), 14630/26822/32683/33683 (MSI services). **Free and therefore
available: 8000, 8002, 8003, 8765, 8788, 9090, 10000, 10001.**
Switchboard took **8002**.

## What Switchboard changed

Additive only; every change is listed in [ROLLBACK.md](ROLLBACK.md) with its
backup. In summary: a new directory `C:\AI\Switchboard`, an added cptr chat
connection and MCP tool server, and an optional step in `start.ps1`. No GGUF was
moved or deleted, no existing connection or MCP server was removed, and
`llama-server`'s launch path is unchanged.

## Notes carried in from earlier sessions

Recorded here because they are constraints on the design, not preferences that
can be re-litigated:

- The discrete GPU is deliberately run in a low-power mode; slow timings are
  measured against that, never reported as a fault.
- The user's local-LLM infrastructure (`Tools\llama-native`, `models\`,
  `start.ps1`) is treated as off-limits for *other* projects. Switchboard is the
  exception only because this project is explicitly about integrating with it,
  and even then it reads far more than it writes.
- Benchmarking the discrete GPU heavily is unwelcome; the benchmark tool
  defaults to one short completion per model and never loads a model unless
  asked with `--switches`.
