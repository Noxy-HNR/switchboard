# The logit backend: what was tried, measured, and concluded

**Result: it works, it is off, and it stays off.** It is implemented and
tested, but its probabilities are not calibrated, and once the System One
Adapter backend was fixed, the adapter beat it on the task judgments that route
every request, on the same machine and fixtures.

## The idea

A System One decision is a bounded choice. Asking a local model to *write* that
choice as JSON makes it generate a hundred tokens to say one word. Instead, ask
the question so that the answer is the **next token**, generate exactly one
token with `logprobs`, and read the probability mass over the candidate labels.

llama.cpp build 11041 supports this directly: `logprobs: true` with
`top_logprobs: N` maps to its `n_probs`, returning the top N candidates for each
generated token. With `max_tokens: 1` the model does one forward pass over the
prompt and stops.

Three practical details decided the implementation:

* **Thinking must be off.** The resident Qwen3.5 models reason by default, so
  the first token would be the start of a thought, not the answer. Both
  `chat_template_kwargs: {"enable_thinking": false}` and
  `reasoning_effort: "none"` are sent.
* **Labels must be single tokens.** Choices are asked as `A`/`B`/`C`, Nouls as
  `yes`/`no`, Scores as digits. Matching is case-insensitive on the stripped
  token, with a first-character fallback for tokenisers that split a label.
* **Invalid mass is recorded, not hidden.** Probability that lands on tokens
  which are not valid labels is measured and reported, then the remainder is
  renormalised. A question where the model spends its mass elsewhere is a
  question the backend answered badly, and the number says so.

The backend reads the same shared state every backend sends
(`decisions/state.py`), rendered as `path:` blocks, and asks the same question
text (`decisions/schema.py`), so its answers are directly comparable.

## What was measured

On this machine, with `Qwen3.5-9B-UD-Q6_K_XL` resident, over the 47 task and 17
step fixtures in `evals/fixtures/`, comparing all three local options with
`switchboard eval --backends local,logit,rules`. Each backend runs in a chain of
its own, so the rules fast path cannot answer on another backend's behalf.

**2026-09-23** (`reports/eval-20260923-043737`), the final code:

| backend | task agreement | step agreement | p50 task latency | p95 |
| --- | ---: | ---: | ---: | ---: |
| **local (System One Adapter)** | **0.681** | 0.882 | 1277 ms | 1334 ms |
| logit | 0.426 | **0.941** | 1532 ms | 1688 ms |
| rules | 0.532 | 0.706 | 0.3 ms | 0.7 ms |

Per question:

| check | local | logit | rules |
| --- | --- | --- | --- |
| domain | 38/43 | **39/43** | 34/43 |
| difficulty | **37/47** | 36/47 | 35/47 |
| tool need | **41/42** | 39/42 | 38/42 |
| ambiguity | **43/44** | 27/44 | 41/44 |
| step outcome | 15/17 | **16/17** | 14/17 |
| step recommendation | **17/17** | **17/17** | 15/17 |

The one step fixture between the adapter and the logit backend is deliberately
contradictory: the output says "Build finished." and the exit code is 1. The
adapter believed the text; Switchboard's exit-code rule overrode it in code, so
its recommendation (`retry`) was still the expected one. An earlier run on
2026-09-22 (`eval-20260922-192216`, before `step.attempt` left the state)
scored the adapter 0.660 on tasks and 0.941 on steps, so one fixture either way
is the resolution of a 17-fixture set.

**2026-09-21**, the run the earlier conclusion rested on:

| backend | task agreement | step agreement |
| --- | ---: | ---: |
| logit | 0.596 | 0.882 |
| rules | 0.532 | 0.706 |
| local (System One Adapter) | 0.340 | 0.588 |

## Why the ranking reversed

**The 2026-09-21 adapter numbers measured a broken backend.** The adapter ran
in its structured mode, where the questions and criteria exist only as
`description`s inside the JSON schema. llama.cpp uses `response_format` purely
as a sampling grammar and never shows the schema to the model, so the model was
filling in a JSON shape without ever being told what the questions were. A
0.34 agreement is what that looks like. A hand-written rubric later patched the
prompt; the current code instead runs the adapter in its prompted mode
(`structured_outputs=False`), where the adapter puts the schema, questions
included, into its own system message, and the grammar still guarantees the
reply parses. With the questions visible, it went from worst to best.

**The logit backend lost ground on ambiguity.** Between the two runs the
question text was rewritten to follow the TypeSafe guidance (name the state
paths, describe situations, state premises) and the state lost fields no
question named. The adapter improved. The logit backend sees the same
instructions and criteria word for word, one question per request, but has to
commit to its first token without working through them, and on the new
ambiguity question it put most of its mass on "yes" for clear system requests
("What is listening on port 8002 right now?" scored 0.82). One question text
now has to serve all backends, and it serves the backend that can work through
the criteria before it answers.

## The conclusions

**It stays off.** The brief allows the experimental backend only if its
probability behaviour can be validated, and it cannot:

* its confidence values are not validated as probabilities. Over the 47 task
  fixtures in `eval-20260923-043737` its domain confidence has a median of
  0.98 (24 of 47 at 0.98 or above) but goes as low as 0.23, with 17 of 47
  below 0.9. A next-token distribution after a well-posed question is often
  sharply peaked, and nothing here shows that a stated 0.9 is right 90% of the
  time, so the values can rank answers but say nothing proven about "how
  likely am I to be right";
* its decisions carry `calibrated=False`, so even if it is turned on, the policy
  never weighs its distributions and uses only the point answers;
* it no longer wins on accuracy either: on tasks, the judgments that route
  every request, it trails the adapter by 0.255 (0.426 vs 0.681), and its
  one-fixture edge on steps changes no recommendation.

**The configured order is `jev -> local -> rules`.** Jev is the one backend
trained for calibrated probabilities, so it goes first when a key is set. The
adapter backend is the best local judgment measured here. The rules engine is
the floor that cannot fail, and the fast path uses it to settle easy requests
in under a millisecond, so the adapter's ~1.3 s is spent only where the answer
could change the route.

**What would change the conclusion.** A calibration check, not an agreement
score: bucket the backend's stated probabilities against labelled outcomes on a
larger fixture set and show that 0.8 means right about 80% of the time. Until
then the backend is one config line away (`decisions.logit.enabled = true`) for
experiments like the eval below; putting it in live routing would also mean
listing it in `decisions.backends`. Nothing in the code or docs calls its
numbers calibrated.

## Reproducing it

```bash
cd /c/AI/Switchboard
SWITCHBOARD_DECISIONS__LOGIT__ENABLED=true uv run switchboard eval --backends local,logit,rules
```

The environment override turns the logit backend on for that process only.
(In PowerShell, set `$env:SWITCHBOARD_DECISIONS__LOGIT__ENABLED = "true"` first;
it then lasts for the session, so remove it afterwards with
`Remove-Item Env:SWITCHBOARD_DECISIONS__LOGIT__ENABLED`.)
llama-server must be up with a model loaded. The report lands in `reports/` as
both JSON and Markdown, and every disagreement is listed with the observed
values so a wrong fixture label can be told apart from a wrong answer.
