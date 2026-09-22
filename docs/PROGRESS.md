# Resumption record

2026-09-22: baseline was 351 passing offline tests, existing integration and
live reports from 2026-09-21. Baseline reports are historical; final audit is
in progress. User approved dashboard switching, redacted cloud summaries for
all topics, additive cptr/start.ps1 integration, and **9B-only GPU tests**.

Audit changes: fixed authorization/JSON credential redaction and truncation;
cloud dump guard/shared text budget; default previews off; experimental logit
backend disabled (candidate probability semantics not sufficiently validated);
Jev/local/rules default; explicit proxy model choices fail instead of silently
substituting; explicit choices bypass semantic providers; checked residency
inside generation reservation; shared switch/generation lock; manual/direct
busy checks; streaming disconnect/error privacy fixes; browser origin/JSON/body
boundary; safer start/stop process identity. New tests in test_hardening.py.

Next: run/fix suite, verify clean install, start 9B+dashboard (existing launch
path) and Switchboard, run live validation/eval/benchmark/MCP/doctor/privacy
checks, inspect cptr integration, correct docs/acceptance, commit source.
Do not load either 27B for testing. No valid Jev key was configured.
