# Resumption record

**2026-09-23 - complete.** 575 offline tests pass; 23/23 live checks, 11/11
integration checks and `doctor` (0 failed) pass against the final code with the
9B resident. See [ACCEPTANCE.md](ACCEPTANCE.md).

User decisions still in force: dashboard switching, redacted cloud summaries
for all topics, additive cptr/start.ps1 integration, **9B-only GPU tests**
(neither 27B was ever loaded), and do not disturb a live Lecture Notes
recording.

What happened after the 2026-09-21 baseline (351 tests):

1. **Audit hardening** (`c870963`): explicit model choices fail instead of
   substituting, residency re-checked under the generation reservation, one
   switch lock for manual and automatic switches, streaming disconnect
   handling, the localhost browser boundary, strict config validation,
   PID-identity checks in the stop script, previews off by default, pasted dumps
   minimized instead of refusing Jev.
2. **TypeSafe review** (30 findings; `e1b6af2`, `1aab9d9`): one shared state for
   every backend holding only fields the questions name; question text
   rewritten as situations; the adapter backend in prompted mode (it had been
   showing the model no questions at all); calibrated distributions used by the
   policy as expected quality; Jev step judgments of machine output handed to a
   local backend; SDK 0.7.1 and adapter 0.2.1; minimizer coverage and
   placeholder wording.
3. **Found while verifying** (`3ee0593`, `1aab9d9`, `dcedf22`): `mypy` clean
   (`3ee0593`); privacy regexes that were quadratic on adversarial input (300k
   spaces took 127 s of event-loop time) made linear (`1aab9d9`); token counts
   no longer redacted as credentials, and switch reasons no longer say a task
   "is above" a ceiling it sits at (`dcedf22`).
4. **Adversarial review** (`4a861c3` and the docs after it): agents checked
   every doc claim against the code and hunted for defects, and each finding
   had to survive an independent attempt to refute it. Fixed: a repeated
   cancellation could cancel a switch mid-restart and release the lock; a
   non-integer `max_tokens` 500'd; the context-fit check sized the answer from
   the wrong difficulty; the eval simulation dropped calibrated distributions;
   JWT redaction was still quadratic; short one-time codes under a `token`
   name escaped redaction; `PASS`/`PWD` names, tilde fences, the zsh prompt
   and over-deep JSON were missed. `step.attempt` left the state. About 35 doc
   statements were corrected.
5. **Re-measured live on the final code**: the adapter backend is the best
   local judgment on tasks (0.681, vs logit 0.426 and rules 0.532), so the chain
   stays `jev -> local -> rules` and logit stays off.

6. **Housekeeping and a Settings page** (2026-09-23): the cptr config backup
   (it held a plaintext key) went to the Recycle Bin; the stale
   `Qwen3.5-9B-Q4_K_M.gguf` entry left `C:\AI\Config\models.json` (backed up
   first); the workspace integration edits were committed on the
   `switchboard-integration` branch of the `C:\AI` repo; the stack was
   restarted through `C:\AI\start.ps1` so it no longer depends on a tool
   session. The dashboard gained a Settings panel: live settings saved to
   `config/local.toml`, keys saved to `.env` and never returned.

Open, and not Switchboard's to decide:

- A TypeSafe API key, to live-test Jev (status BLOCKED_EXTERNAL). It can now be
  entered and tested on the dashboard's Settings panel.
- Merging (or pushing) the `switchboard-integration` branch of the `C:\AI` repo.
