# Acceptance

Every requirement, its status, and the evidence. Verified on this machine on
**2026-09-23** against the final code (commit `4a861c3`; later commits change
only documentation), with `Qwen3.5-9B-UD-Q6_K_XL` resident on llama-server, the
dashboard running, and no TypeSafe API key configured. The first pass was on
2026-09-21; everything below was re-run after the audit, the TypeSafe review
and an adversarial review of the result, and the numbers are the final run's.

Statuses: **PASS** (verified here) · **BLOCKED_EXTERNAL** (built and tested, but
final verification needs something unavailable) · **NOT_APPLICABLE** (discovery
showed it does not apply to this installation).

Reports cited below are in `reports/`: `live-20260923-043849` (23 live checks),
`eval-20260923-043737` (backend comparison), `benchmark-20260923-043907`
(resident 9B), `benchmark-20260922-192355` (a real 9B reload) and
`integration-validation.json` (cptr and MCP integration).

| # | Requirement | Status | Evidence |
| --- | --- | --- | --- |
| 1 | A maintainable implementation, not a scaffold | PASS | 32 modules under `src/switchboard/`, 575 tests, `mypy` reports no issues in any module, docs. No TODOs standing in for behaviour. |
| 2 | Installs into a clean project-local environment | PASS | CLEAN_INSTALL_EVIDENCE |
| 3 | Automated tests pass | PASS | `uv run pytest -q` → **575 passed**. Offline: every test uses mock transports, so the suite passes with the stack stopped. |
| 4 | Switchboard starts | PASS | `scripts/start.ps1` → "Switchboard is ready"; the PID file records PID and start time, and `scripts/stop.ps1` checks both plus the command line before stopping anything. |
| 5 | Health/status endpoints work | PASS | `GET /health` → `{"status":"ok","mode":"active","upstream_healthy":true,"resident_model":"qwen3.5-9b"}`; `switchboard doctor`: 16 checks, 0 failed. |
| 6 | `/v1/models` exposes `switchboard-auto` | PASS | Live `models`: `switchboard-auto`, `-fast`, `-deep`, `-medical` plus the three real models. |
| 7 | Non-streaming chat reaches a real local model | PASS | Live `easy_auto` and `coding_auto` answered by the 9B (HTTP 200, real `usage`). |
| 8 | Streaming works incrementally | PASS | Live `incremental_stream`: **80 content chunks**, first at 369 ms, spread over 4.08 s. Not buffered. |
| 9 | Tool definitions survive forwarding | PASS | Live `tools_request` and `tool_definition_survived`: a request with an `add_numbers` tool got back a real `tool_calls` entry naming it. |
| 10 | Explicit model selection bypasses routing | PASS | Live `explicit_bypasses_semantics`: `X-Switchboard-Backend: bypass`. Live `unknown_model_rejected`: an unknown name is a 404; an explicit model that cannot be loaded is a 409 (`tests/test_proxy.py`), never a silent substitute. |
| 11 | Routing combines semantic judgments with real facts | PASS | Every reason names both, e.g. "Switching to qwen3.8-27b: this coding task at difficulty 2.0 sits at qwen3.5-9b's 2.0 ceiling. Expected quality 0.50 -> 0.78 is worth the ~20s load." (eval switching trace). With a calibrated backend the policy scores the full domain and difficulty distributions and names a split that mattered. |
| 12 | Works without a TypeSafe key | PASS | No key configured throughout. Jev reports `unconfigured`; the chain uses the local adapter backend and the rules floor. |
| 13 | Jev live-tested if a key is available | BLOCKED_EXTERNAL | No valid key exists on this machine. On 2026-09-21 (typesafe-sdk 0.7.0, before the shared-state payload of `e1b6af2`) a request reached `api.typesafe.ai` and returned a genuine `401 TypeSafeAuthenticationError` with a request id, proving URL, headers and error mapping for that version; no report records it, and the current payload has not reached the live API. Answer parsing, the shared state, privacy minimization and failure classification are covered by 48 mocked tests against `typesafe-sdk` 0.7.1. |
| 14 | A Jev failure demonstrates fallback | PASS | Live `jev_outage_fallback`: a synthetic 503 from the Jev transport fell through to the local backend running on the real 9B. Auth errors and refusals do not open the breaker. |
| 15 | Secret-redaction/privacy tests pass | PASS | 168 privacy and hardening tests, including linear-time bounds on adversarial input. Live `privacy_canaries_not_persisted`: the canary secret and marker sent through the proxy were found nowhere in the database or logs. Read-only scan of the live database: 168 rows, 997 text cells, **0 secret-shaped cells**, no multi-line cells; the longest text cell is a 949-character routing decision. |
| 16 | Conversation/model stickiness works | PASS | `tests/test_router.py` (sessions accumulate turns, model and running difficulty) and `tests/test_policy.py::test_a_hard_conversation_does_not_fall_back_on_one_short_turn`. |
| 17 | Hysteresis prevents pointless switching | PASS | Eval switching simulation over 47 fixtures, replaying each backend's real decisions: rules backend **4 switches with the guards, 7 without**; adapter backend 3 and 3 (every switch it proposed cleared the guards on quality). `tests/test_policy.py` covers each guard separately. |
| 18 | Direct llama.cpp usage still available | PASS | Live `direct_llama_still_works`: `POST http://127.0.0.1:10000/v1/chat/completions` answered with Switchboard running; `resident_unchanged` afterwards. |
| 19 | Existing MCP integrations intact | PASS | `scripts/validate_integrations.py`: 11/11, including cptr's pre-existing connections and tool servers preserved, the additions present in its runtime database, all three Switchboard MCP tools listed, and `switchboard_status` and `switchboard_route` called. All three tools are called by the live `mcp_switchboard_*` checks. |
| 20 | GGUF files untouched | PASS | All three unchanged, modified 2026-08-22/23. Nothing was moved, copied or deleted. Only the 9B was ever loaded. |
| 21 | Existing startup behaviour usable | PASS | `doctor`'s `integration.start.ps1` check confirms `C:\AI\start.ps1` contains the block that runs `scripts\start.ps1`; the block is bracketed, guarded by `Test-Path`, and wrapped in try/catch so it cannot stop the stack. No automated check runs `C:\AI\start.ps1` end to end; the dashboard was started exactly as it starts it. |
| 22 | `doctor` gives meaningful diagnostics | PASS | 16 checks, 0 failed, 4 warnings, each naming what to do: a stale `Qwen3.5-9B-Q4_K_M.gguf` entry in `Config/models.json` with no file (reported twice), no Jev key, and low free RAM. |
| 23 | Eval/benchmark tooling runs | PASS | `switchboard eval --backends local,logit,rules` over 47 task + 17 step fixtures; `switchboard benchmark` measured the 9B at 20.5 tok/s generation and 90.7 tok/s prompt from llama.cpp's own counters. |
| 24 | Switching cost represented in routing | PASS | A real reload of the 9B through the dashboard measured **11.0 s** and was written to the store as its `load_s`; the policy reads measured load times back as the switch cost. medgemma-27b's 27.6 s load comes from `llama.stderr.log` (2026-09-18); qwen3.8-27b's is a size-derived estimate (1.7 s/GB). A real switch replaces either with Switchboard's own measurement. |
| 25 | Documentation reflects the actual installation | PASS | `docs/DISCOVERY.md` records the machine as found. Every number in the docs is cited to a report or a command, and every factual claim in the docs was checked against the code by an adversarial review, with the confirmed errors corrected. |
| 26 | Rollback instructions and backups exist | PASS | `docs/ROLLBACK.md`; backups in `backups/` for `models.json` (one per Switchboard process, taken before its first switch), `cptr-config.toml` and `start.ps1`; both integration scripts take `-Remove`. |
| 27 | One simple normal way to start and use it | PASS | `scripts\start.ps1`, then `switchboard-auto` at `http://127.0.0.1:8002/v1`. `C:\AI\start.ps1` runs it too. |

## The decision backends

| Requirement | Status | Evidence |
| --- | --- | --- |
| TypeSafe Jev as the primary judgment | BLOCKED_EXTERNAL | Implemented per the live TypeSafe docs and reviewed against them: one batched call per judgment, a state holding only fields the questions name, question text written as situations, pasted machine output replaced by a described placeholder before anything leaves, calibrated distributions used by the policy. Needs a key for a live answer. |
| Local/private fallback | PASS | The official System One Adapter (0.2.1) over the resident model, in prompted mode with the schema as a grammar. Best of the local options on the task judgments that route every request: **0.681 task agreement**, 0.882 on steps, p50 1.3 s. |
| Deterministic fallback | PASS | The rules backend cannot fail and is forced to the end of every chain: 0.532 / 0.706 agreement in under a millisecond. |
| Experimental logit backend stays off unless validated | PASS (off) | Implemented and tested, disabled by config. Its probabilities could not be validated as calibrated, and on tasks it measured well below the adapter (0.426). See [EXPERIMENT-LOGITS.md](EXPERIMENT-LOGITS.md). |

## How it was verified

Beyond the tests and live checks, the final state went through an adversarial
review: independent agents checked every factual claim in the docs against the
code and hunted for defects in the session's code changes, and each finding
had to survive a second agent's attempt to refute it. The confirmed defects
were fixed with regression tests, among them a switch that a repeated
cancellation could interrupt mid-restart, a non-integer `max_tokens` that
turned into a 500, and privacy patterns that were quadratic on crafted input
or let short one-time codes through. Confirmed doc errors were corrected.

## Requirements that discovery retired

| Requirement | Status | Why |
| --- | --- | --- |
| Open WebUI integration | NOT_APPLICABLE | Not installed anywhere on this machine. Steps documented in [INTEGRATION.md](INTEGRATION.md) if it ever is. |
| Open Terminal MCP preserved | NOT_APPLICABLE | No such server exists here. The MCP servers that do exist were preserved. |
| Integration with `C:\AI\start-ai.ps1` | NOT_APPLICABLE | No such file. The real launcher is `start.ps1`, and that is what was integrated. |
| llama.cpp router mode as the switch mechanism | NOT_APPLICABLE (by choice) | Supported by this build, but using it would change the launch path the brief asks to preserve. Detected and reported; rationale in [DEPENDENCY-VERIFICATION.md](DEPENDENCY-VERIFICATION.md#why-router-mode-is-not-used). |
| cptr's authenticated web UI | BLOCKED_EXTERNAL | Checking the Switchboard connection inside cptr's UI needs your login; no session was fabricated. Its configuration and runtime database were verified directly. |

## Things worth knowing

* **Generation speed varies.** The 9B measured 38.0 tok/s on 2026-09-21, 40.5
  tok/s at 13:39 UTC on 2026-09-22, 20.3-20.6 tok/s at 19:23 UTC the same day
  and 20.5 tok/s at 04:39 UTC on 2026-09-23, the later runs with free RAM at
  about 0.5 GB of 15.4 GB. The policy reads the latest measurement, so routing
  follows the machine as it is.
* **Only the 9B was ever loaded**, as agreed. The 27B costs come from existing
  logs and one size-derived estimate: qwen3.8-27b's speed from the tuning log
  and its load time from file size (1.7 s/GB), medgemma-27b's speeds and 27.6 s
  load from `llama.stderr.log` (2026-09-18). They become Switchboard's own
  measurements the first time a switch to them happens.
* **Non-streaming hard requests stay on the fast model.** A streamed answer
  starts arriving immediately, so slow generation is discounted; a
  non-streaming caller waits for the whole thing and the arithmetic reflects
  that. Deliberate, and adjustable via `policy.generation_wait_weight`.
* **Processes started from a tool session end with it.** During verification
  the Claude app restarted and took the dashboard, llama-server and Switchboard
  with it, because they had been launched from its shells; they were started
  again the same way. Started normally through `C:\AI\start.ps1`, they run
  independently.
* **Three changes are uncommitted in the `C:\AI` workspace repo**
  (`start.ps1`, `Config/models.json`, and a `.gitignore` line for the nested
  `Switchboard/` repo). They are left for you to review rather than committed
  on your behalf. `Config/llama-native.pid` there is the dashboard's runtime
  file, not a Switchboard change.
