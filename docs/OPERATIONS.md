# Operational readiness — 2026-09-26

Switchboard is configured for normal use on this Windows machine. The normal
entry point is `C:\AI\start.ps1`: it checks the resident llama.cpp server, starts
cptr, and starts Switchboard. `-NoUi -NoPause` performs the same startup without
opening interfaces or waiting for Enter. The startup path was exercised with
Switchboard stopped and returned successfully after all services became healthy.

## Use and diagnosis

| Purpose | Address or command |
| --- | --- |
| Start the complete stack | `C:\AI\start.ps1` |
| cptr | `http://127.0.0.1:8000` — choose connection **Switchboard**, model **switchboard-auto** |
| Switchboard dashboard | `http://127.0.0.1:8002/dashboard` |
| OpenAI-compatible API | `http://127.0.0.1:8002/v1` |
| API documentation | `http://127.0.0.1:8002/docs` |
| MCP | `http://127.0.0.1:8002/mcp/` |
| Diagnose | `C:\AI\Switchboard\scripts\doctor.ps1` |
| Stop only Switchboard | `C:\AI\Switchboard\scripts\stop.ps1` |
| Existing direct model API | `http://127.0.0.1:10000/v1` |

Switchboard is **active**, using `jev -> local -> rules` after the rules fast
path. Jev is configured and answered real synthetic task and step judgments.
The local backend is TypeSafe's System One Adapter on the resident model.
The experimental logit backend is absent from the chain and does not route
production requests, regardless of its saved enabled flag.

## Current model policy

| Model | Configured context | Treatment |
| --- | ---: | --- |
| Qwen3.5 9B Q6_K_XL | 180,224 | Resident daily driver; all GPU validation used this model |
| Qwen3.8 27B IQ3_M | 20,480 | Existing reasoning profile; automatic switches subject to guards |
| MedGemma 27B Q4_K_M | 32,768 | Existing medical profile; no tool support; subject to guards |
| Huihui Qwen3.8 27B IQ3_XXS MTP | 24,576 | Explicit/manual selection only until capability and speed validation |

The new Huihui model is in `routing.auto_exclude` in `config/local.toml`.
It remains discoverable and may be requested explicitly. No capabilities or
benchmark measurements have been invented for it. `doctor` therefore retains
its two model warnings. The new file and the changed 9B context/KV settings
were discovered in the parent stack; this pass did not edit `models.json`.

The local RAM guard is `block`. This machine had less than 1 GB of free host
RAM during validation. Automatic switches to a model whose estimated host-RAM
needs do not fit are blocked. This is a conservative estimate, not a memory
reservation or an OOM guarantee. Explicit model requests remain user overrides.

## Evidence from this pass

- Automated tests: **637 passed** (`uv run pytest -q`).
- Types: `uv run --with mypy mypy src/switchboard --ignore-missing-imports`
  found no issues in 34 source files.
- Lint: `uv run --with ruff ruff check src tests --select F,B,RUF059` passed.
- `scripts/validate_live.py`: **23/23 checks passed**, repeated after the final
  settings and restart, in `reports/live-20260926-200955.json`: real auto chat, explicit bypass, streaming,
  tool calls, Jev task/step judgments, simulated Jev 503 with real local fallback,
  MCP, direct llama.cpp, unchanged resident, and no persisted privacy canaries.
  The final easy request took 613 ms; coding took 2.06 s. Streaming delivered
  80 content chunks, first at 470 ms, over 2.29 s. The earlier cold first request
  took 20.9 s: startup health alone does not guarantee warm inference latency.
- `scripts/validate_integrations.py`, run with cptr's Python/MCP installation:
  **10/10 current checks passed** in
  `reports/integration-validation-20260926-200441.json`. cptr HTTP is now up;
  the existing direct connection and tool-server entry remain present, and the
  Switchboard additions match cptr's runtime database.
- Doctor: **16 checks, 0 failures, 3 warnings** in
  `reports/doctor-20260926-150752.json`. Warnings are the unprofiled/unmeasured
  Huihui model and low free host RAM, addressed operationally by the guards above.
- Resident benchmark: **31.133 generated tokens/s**, with 64 output tokens,
  in `reports/benchmark-20260926-200524.json`. No model reload or switch.

The evaluation used 47 task and 17 step fixtures per backend. Agreement means
all applicable engineering expectations for a fixture matched; these labels
are not objective ground truth or a guarantee of answer quality.

| Backend | Task agreement | Step agreement | Task p50 / p95 | Step p50 / p95 |
| --- | ---: | ---: | ---: | ---: |
| Jev | 63.8% | 66.7% | 305 / 511 ms | 245 / 305 ms |
| Local adapter | 74.5% | 88.2% | 1,093 / 1,468 ms | 1,102 / 1,389 ms |
| Rules | 53.2% | 70.6% | 0.3 / 0.7 ms | <0.1 / 0.1 ms |

Evidence: `reports/eval-20260926-200752.json` and `.md`. The two Jev step
"errors" are intentional privacy skips for machine-output evidence; local and
rules backends handle those in the chain. The estimated Jev input cost of this
evaluation was $0.002771. No provider ordering was changed on the basis of one
fixture run. The local results support retaining it as the private fallback.

## Scope and remaining external validation

**PASS:** normal startup, health, real 9B chat and streaming, tools, fallback,
Jev, MCP, current cptr configuration/HTTP, tests, diagnostics, benchmark/eval,
and preservation of the direct llama.cpp path.

**BLOCKED_EXTERNAL:** cptr's authenticated browser workflow requires the user's
login; no session was fabricated. Real 27B loading and inference remain outside
the user's approved 9B-only GPU test scope. The recording guard has automated
coverage but was not exercised by disrupting a live recording. Historical cptr
backup comparison is unavailable because that backup was previously removed
for containing a credential; current entries were checked directly instead.

**NOT_APPLICABLE:** Open WebUI integration; this installation uses cptr.

## Changes and rollback

The startup hang was caused by waiting on a native pipeline after a detached
service's launcher exited. The master script and integration template now wait
on the launcher process with file-backed output and a 60-second bound. The
normal interactive invocation remains available.

Backups: `backups/operation-20260926-150303/start.ps1` and `local.toml`.
Restore the former to `C:\AI\start.ps1` to undo the startup changes. Restore
the latter to `config/local.toml` and restart Switchboard to undo the new model
exclusion and RAM guard. General integration rollback is in [ROLLBACK.md](ROLLBACK.md).

No GGUF file was changed, moved, or deleted. No 27B model was loaded. The
existing llama-server process was preserved throughout the operational pass.
