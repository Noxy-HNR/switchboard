# The logit backend: what was tried, measured, and concluded

**Result: it works, it is enabled, and its confidence numbers are not
calibrated.** Both halves of that sentence matter.

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

## What was measured

On this machine, with `Qwen3.5-9B-UD-Q6_K_XL` resident, over the 47 task and 17
step fixtures in `evals/fixtures/`, comparing all three local options
(`switchboard eval --backends local,logit,rules`, 2026-09-21):

| backend | task agreement | step agreement | p50 task latency | p95 |
| --- | ---: | ---: | ---: | ---: |
| **logit** | **0.596** | **0.882** | 2312 ms | 3571 ms |
| rules | 0.532 | 0.706 | 0.6 ms | 1.9 ms |
| local (System One Adapter) | 0.340 | 0.588 | 2095 ms | 2416 ms |

Per question:

| check | logit | rules | local |
| --- | --- | --- | --- |
| domain | **40/43** | 34/43 | 22/43 |
| difficulty | **38/47** | 35/47 | 35/47 |
| tool need | **42/42** | 38/42 | 40/42 |
| ambiguity | 36/44 | **41/44** | 40/44 |
| step outcome | **17/17** | 14/17 | 15/17 |
| step recommendation | **17/17** | 15/17 | 14/17 |

The values are also well separated in the way a routing policy needs: a
greeting scores difficulty 0.06, a definition 0.83, a scheduler design 2.84.

## The conclusions

**It is better than the official adapter here, at the same cost.** That was the
surprise. The System One Adapter misread the domain on half the fixtures, which
is what a 9B does when asked to fill in a JSON schema in one pass without
thinking. Reading the first token asks the model a much easier question. The
adapter remains implemented, tested and one config line away, because it is the
official path and would likely win on a model that follows JSON instructions
better.

**Its probabilities are not calibrated, and the code says so.** The confidence
values sit at 0.98-0.99 almost everywhere, because a next-token distribution
after a well-posed question is sharply peaked. That makes them useful for
*ranking* and useless as "how likely am I to be right". So:

* the routing policy reads `domain` and `difficulty` and never thresholds the
  confidence of this backend;
* the backend records the invalid-mass fraction on every answer;
* nothing in the code or the documentation calls these calibrated.

This is the difference between this backend and Jev, which is trained for
calibrated probabilities. If Jev is configured, it goes first.

**It is not free.** 2.3 seconds per task decision, because four questions are
four forward passes over the prompt. That is why the fast path exists: the rule
engine settles easy requests in under a millisecond, and this backend is only
consulted when the answer might actually change the routing. In practice the
measured decision latency on routine traffic is 1-9 ms, and only genuinely hard
or medical requests pay the seconds.

## Reproducing it

```bash
cd C:\AI\Switchboard
uv run switchboard eval --backends local,logit,rules
```

Requires llama-server up with a model loaded. The report lands in `reports/`
as both JSON and Markdown, and every disagreement is listed with the observed
values so a wrong fixture label can be told apart from a wrong answer.
