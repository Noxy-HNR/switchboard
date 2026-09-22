# Acceptance

Every requirement, its status, and the evidence. Verified on this machine on
**2026-09-21**, with `Qwen3.5-9B-UD-Q6_K_XL` resident on llama-server, the
dashboard running, and no TypeSafe API key configured.

Statuses: **PASS** (verified here) · **BLOCKED_EXTERNAL** (built and tested, but
final verification needs something unavailable) · **NOT_APPLICABLE** (discovery
showed it does not apply to this installation).

| # | Requirement | Status | Evidence |
| --- | --- | --- | --- |
| 1 | A maintainable implementation, not a scaffold | PASS | 24 modules under `src/switchboard/`, 351 tests, docs. No TODOs standing in for behaviour. |
| 2 | Installs into a clean project-local environment | PASS | `scripts/setup.ps1` → `uv sync` into `C:\AI\Switchboard\.venv`. Nothing global changed. |
| 3 | Automated tests pass | PASS | `uv run pytest -q` → **351 passed**. Offline: every test uses mock transports. |
| 4 | Switchboard starts | PASS | `scripts/start.ps1` → "Switchboard is ready", PID recorded, health answered. |
| 5 | Health/status endpoints work | PASS | `GET /health` → `{"status":"ok","resident_model":"qwen3.5-9b"}`; `GET /status` returns upstream, models, providers, host. |
| 6 | `/v1/models` exposes `switchboard-auto` | PASS | Live: `switchboard-auto`, `-fast`, `-deep`, `-medical` plus the three real models, at `/v1/models` **and** `/models`. |
| 7 | Non-streaming chat reaches a real local model | PASS | Live completion through the 9B, `timings.predicted_per_second = 38.7`. |
| 8 | Streaming works incrementally | PASS | Real socket: **202 chunks**, first at 0.597 s, last at 5.546 s. Not buffered. |
| 9 | Tool definitions survive forwarding | PASS | Live request with a `list_dir` tool → `finish_reason: tool_calls` and a real tool call returned. |
| 10 | Explicit model selection bypasses routing | PASS | `{"model":"qwen3.5-9b"}` → reason: "named by the request and is already loaded"; no routing performed. |
| 11 | Routing combines semantic judgments with real facts | PASS | Every reason names both, e.g. "a coding task at difficulty 0.8 sits within its 2.0 ceiling. qwen3.8-27b would cost ~20s to load for +0.04 expected quality." |
| 12 | Works without a TypeSafe key | PASS | No key configured throughout. Jev reports `unconfigured`; the chain falls through to logit and rules. |
| 13 | Jev live-tested if a key is available | BLOCKED_EXTERNAL | No valid key exists on this machine. The transport **was** exercised against the live API: a request reached `api.typesafe.ai` and returned a genuine `401 TypeSafeAuthenticationError` with a `request_id`, proving URL, headers, payload shape and error mapping. Answer parsing is covered by 42 mocked tests. |
| 14 | A Jev failure demonstrates fallback | PASS | That real 401 fell through: `jev FAILED (auth) → logit → rules ok`, request still answered. Auth errors correctly did not open the breaker. |
| 15 | Secret-redaction/privacy tests pass | PASS | Unit tests plus a scan of the live database: 44 rows, 7 tables, **zero secret-shaped strings**; longest free text was Switchboard's own routing reason. |
| 16 | Conversation/model stickiness works | PASS | `tests/test_router.py` (sessions accumulate turns, model and running difficulty) and `tests/test_policy.py::test_a_hard_conversation_does_not_fall_back_on_one_short_turn`. |
| 17 | Hysteresis prevents pointless switching | PASS | Eval switching simulation over 47 fixtures: **4 switches with the guards, 7 without**. `tests/test_policy.py` covers each guard separately. |
| 18 | Direct llama.cpp usage still available | PASS | `POST http://127.0.0.1:10000/v1/chat/completions` answered normally with Switchboard running. cptr's original connection is untouched. |
| 19 | Existing MCP integrations intact | PASS | cptr's `user_health` server still present after the edit; Switchboard added alongside. All 44 cptr config keys preserved. |
| 20 | GGUF files untouched | PASS | All three unchanged, mtimes still 2026-08-22/23. Nothing was moved, copied or deleted. |
| 21 | Existing startup behaviour usable | PASS | `start.ps1` parses cleanly after the edit; the added block is bracketed, guarded by `Test-Path`, and wrapped in try/catch so it cannot stop the stack. |
| 22 | `doctor` gives meaningful diagnostics | PASS | 16 checks; live run: 0 failed, 6 warnings, each naming the file or command to fix it. Works with everything stopped. |
| 23 | Eval/benchmark tooling runs | PASS | `switchboard eval` over 47 task + 17 step fixtures across three backends; `switchboard benchmark` measured 38.0 tok/s generation and 85.1 tok/s prompt from llama.cpp's own counters. Reports in `reports/`. |
| 24 | Switching cost represented in routing | PASS | Real switch through the dashboard measured at **15.2 s** and written to the store; the policy reads it back as the switch cost. A cold load of the same model measured 25.2 s. |
| 25 | Documentation reflects the actual installation | PASS | `docs/DISCOVERY.md` records the machine as found, including what was *not* installed. Every number in the docs is cited to a log, a report or a command. |
| 26 | Rollback instructions and backups exist | PASS | `docs/ROLLBACK.md`; backups in `backups/` for `models.json`, `cptr-config.toml` and `start.ps1`; both integration scripts take `-Remove`. |
| 27 | One simple normal way to start and use it | PASS | `scripts\start.ps1`, then `switchboard-auto` at `http://127.0.0.1:8002/v1`. `C:\AI\start.ps1` now does it too. |

## Requirements that discovery retired

| Requirement | Status | Why |
| --- | --- | --- |
| Open WebUI integration | NOT_APPLICABLE | Not installed anywhere on this machine. Steps documented in [INTEGRATION.md](INTEGRATION.md) if it ever is. |
| Open Terminal MCP preserved | NOT_APPLICABLE | No such server exists here. The MCP servers that do exist were preserved. |
| Integration with `C:\AI\start-ai.ps1` | NOT_APPLICABLE | No such file. The real launcher is `start.ps1`, and that is what was integrated. |
| llama.cpp router mode as the switch mechanism | NOT_APPLICABLE (by choice) | Supported by this build, but using it would change the launch path the brief asks to preserve. Detected and reported; rationale in [DEPENDENCY-VERIFICATION.md](DEPENDENCY-VERIFICATION.md#why-router-mode-is-not-used). |
| Experimental logit backend | PASS (enabled) | Validated, not assumed: it beat both alternatives on this machine. Its confidence values are explicitly **not** calibrated and the policy never thresholds them. See [EXPERIMENT-LOGITS.md](EXPERIMENT-LOGITS.md). |

## Things worth knowing

* **`activeModel` changed.** It was `medgemma-27b-it-Q4_K_M.gguf`; it is now
  `Qwen3.5-9B-UD-Q6_K_XL.gguf`, because the 9B was loaded for testing through
  the dashboard, which records the loaded model as the stack default. That is
  the switching mechanism you chose. [ROLLBACK.md](ROLLBACK.md) has the one
  command to put MedGemma back.
* **Only the 9B was ever loaded**, as agreed. The 27B costs in the policy come
  from the tuning log and from a size-derived estimate, both labelled as such;
  they become measurements the first time a switch to them happens.
* **Non-streaming hard requests stay on the fast model.** A streamed answer
  starts arriving immediately, so slow generation is discounted; a
  non-streaming caller waits for the whole thing and the arithmetic reflects
  that. Deliberate, and adjustable via `policy.generation_wait_weight`.
* **Two changes are uncommitted in the `C:\AI` workspace repo**
  (`start.ps1`, `Config/models.json`, plus a `.gitignore` line for the nested
  `Switchboard/` repo). They are left for you to review rather than committed
  on your behalf.
