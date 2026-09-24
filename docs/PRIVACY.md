# Privacy

Two questions, answered separately: what leaves the machine, and what is kept
on it.

## What leaves the machine

**Only the Jev backend sends anything anywhere**, and only when
`TYPESAFE_API_KEY` is set and `privacy.allow_cloud` is true. With no key,
Switchboard is entirely local: the rule backend opens no connection at all, and
the local adapter and the (disabled by default) logit backend talk only to
llama-server at `upstream.base_url` (`127.0.0.1:10000`).

When Jev is enabled, each decision sends one request (the SDK resends that same
request up to `decisions.jev.max_retries` more times, 1 by default, when an
attempt fails in a retryable way; no new data is added). It carries a small JSON
object holding only the fields a question names (`decisions/state.py`, shared
by every backend):

```json
{"request": {"message": "<latest user turn>",
             "previous_message": "<the one before it, when there is one>",
             "has_images": true}}
```

`previous_message` and `has_images` are left out when there is nothing to say.
The system prompt, the client's tool names, the turn number, the session id and
the token estimate are never sent: no question is about them, and the policy
reads them directly on this machine.

A finished agent step sends `task.goal`, `step.action`, `step.result`,
`step.exit_code` and up to three `earlier_attempts`, through the same steps
below. The retry count is not sent: no question is about it, and the
recommendation reads it on this machine. One exception: when the step's **result** is itself pasted
machine output (the usual case for a terminal command), Jev is not asked at
all. Minimizing it would leave the step questions judging a placeholder in
place of the evidence, so the step is declined as `skipped` and judged by a
local backend that can read the real output.

Before any of that is assembled, every string goes through three steps:

1. **Pasted machine output stays on this machine.** Python tracebacks, fenced
   code blocks (backtick or tilde), runs of terminal output (shell prompts,
   log lines, stack frames, compiler diagnostics, directory listings, git
   status, `KEY=value` environment lines), JSON and XML/HTML documents, and PEM
   key material are replaced by a one-line description. As a backstop, any
   paragraph of four or more lines that reads as data rather than sentences
   (unfenced YAML, CSV or code) becomes `[pasted text omitted: N lines]`:

   ```
   Why does this fail?
   [pasted Python traceback omitted: 23 lines]
   ```

   The wording is for a model that reads literally: "pasted ... omitted" says
   the user did attach the content, and the question text tells the model such
   a marker stands for content the assistant will see in full. So a request
   is not scored as ambiguous, or as needing a tool, merely because its
   evidence was withheld from the router.

   The contents never leave; the user's own words around them do, because
   those are what the routing judgments are actually about. The detectors err
   toward removing too much, and a pasted JSON blob is recognised by its shape
   even when it is truncated and would not parse. This step cannot be turned
   off, and it does not depend on redaction - it is checked with redaction
   disabled.
2. **Secrets are stripped** from what remains. `privacy.py` replaces
   credential-shaped spans with typed placeholders: API keys (OpenAI,
   Anthropic, GitHub, AWS, Slack, Google, Browser Use, TypeSafe), bearer
   tokens, `Authorization:`/`Proxy-Authorization:` and `Cookie:`/`Set-Cookie:`
   header lines (a Basic credential is caught only inside such a header), JWTs,
   private key blocks, connection strings with passwords, `key = value` and
   JSON `"key": "value"` pairs whose name looks like a credential (including
   `PASS` and `PWD` names such as `DB_PASS`), `.env`-shaped lines, email
   addresses, US social security numbers (123-45-6789) and card-shaped digit
   runs. One exception is deliberate: a count of up to nine digits under a
   plural *tokens* name, such as `max_tokens: 256`, is routing signal and is
   kept; a singular token name keeps every value redacted, because one-time
   and reset codes are that short. Every pattern runs in time linear in the
   text, so a crafted request cannot stall the service.
3. **One shared budget.** `privacy.max_state_chars` (1500 by default) caps the
   whole state, not each field, in fixed shares: the previous message gets at
   most 300 characters and the current message the rest; for a step, the goal,
   the action and the earlier attempts are each capped and the result gets
   everything they leave. An optional field (the previous message, an earlier
   attempt) that comes out empty is dropped rather than sent as an empty
   string; the message, goal, action and result are always sent, and an empty
   result is kept on purpose because it is itself evidence. Long text keeps
   both ends, because the topic is at the start and the question at the end.

Which *kinds* of content were withheld is recorded in the decision's notes -
"kept local: Python traceback", "redacted: openai-key, email" - and the content
itself is recorded nowhere.

Only the latest turn goes, plus the one before it: not the conversation, not
file contents, not tool definitions or results. A 40-message conversation
contributes its last two user turns.

An earlier version refused to call Jev at all whenever a message looked like a
dump (any code fence, or more than eight lines). That also kept dumps local,
but it withheld most real coding questions from the one calibrated backend, so
it was replaced by the minimization above.

Images are never forwarded: Jev is text-only, so their presence is a boolean.

This is a safety net, not a guarantee. Redaction is pattern-based, so a secret
in a shape nobody anticipated can survive it. If that matters more than routing
accuracy, set `privacy.allow_cloud = false` and the machine stays closed.

### Turning it off

| Want | Setting |
| --- | --- |
| Never use the cloud, even with a key | `privacy.allow_cloud = false` |
| Use the cloud but send less | lower `privacy.max_state_chars` |
| Use the cloud with redaction off (not advised) | `privacy.redact = false` |
| Disable Jev specifically | `decisions.jev.enabled = false` |

## What is kept on this machine

**Keys** live in `.env` in the project root (git-ignored), written there by the
dashboard's Settings page or by hand. The settings API never returns a key: it
reports whether one is set, whether it came from `.env` or the process
environment, and its last four characters. The page takes a key in a password
field and empties it the moment it is sent. A key never goes into
`config/local.toml`, the database or a log. The **Test** button sends TypeSafe
one fixed synthetic sentence, never a real request, and is refused while cloud
decisions are turned off. The Settings API sits behind the same localhost
boundary as everything else (below), so a web page cannot write a key or a
setting. Lecture Notes (`C:\AI\Projects\lecture-notes`) reads the TypeSafe key
from this same file for its own Jev features (exam hints and search ranking);
what it sends is described in its README, under "Jev", and its calls are not
counted in the usage line on the key's row.

On Windows, a key saved through Settings is written to a random temporary file
whose ACL is restricted to the current user, SYSTEM and administrators before
the key bytes are written, then atomically replaces `.env`. This matters on this
machine because an ordinary file under `C:\AI` otherwise inherits read access
for other local users. The existing `.env` ACL was tightened on 2026-09-23;
its previous ACL descriptor, without key contents, is in `backups/`.

**Settings** saved from the dashboard live in `config/local.toml`
(git-ignored).

The SQLite database at `data/switchboard.sqlite3`. Its rules are enforced in
code, not by convention:

* **No column can hold a prompt.** Requests are identified by a SHA-256 prefix.
  The store rejects a "hash" longer than 64 characters or containing a newline,
  so a caller cannot smuggle text through the hash column either.
* **No previews by default.** `privacy.store_previews` is **off**: the
  database holds a hash of each request and nothing of its text. Turning it on
  keeps a redacted excerpt of `privacy.preview_chars` characters (80 by
  default) for debugging. The router redacts it, and the store then redacts
  and truncates it again regardless of what it was given. `preview_chars` can
  be at most 200: a larger value is rejected when the config loads, and the
  store clamps to 200 anyway.
* **Agent steps arrive hashed.** Goals and actions are hashed by the router
  before the store sees them.
* **Nothing upstream says is kept.** When llama-server or a decision provider
  answers with an error, the database records the status or the exception
  type - never the body, which can echo the request back.

**History is not kept forever.** Routing rows, decision attempts, switches,
step checks and idle sessions older than `observability.retention_days` (30 by
default; Settings > Data) are deleted automatically once a day, and once a
minute after start-up. Speed and load-time measurements are kept, since they
contain no request content and routing depends on them.

To keep a short redacted excerpt for debugging:

```toml
[privacy]
store_previews = true
```

### What the logs hold

`logs/switchboard.err.log` holds everything the service logs; `.out.log` stays
empty. At INFO level that is uvicorn's startup lines, Switchboard's own startup
line and warnings, MCP session lines, and one httpx line per HTTP call
Switchboard makes (to llama-server, the `C:\AI` dashboard or a decision
provider), giving only the method, URL and status, for example
`HTTP Request: POST http://127.0.0.1:10000/v1/chat/completions "HTTP/1.1 200 OK"`.
Request and response bodies are not logged.

The Jev backend builds its error messages from provider metadata: the exception
type, the status code and the request id. Three exceptions: an error the SDK
raises while building a request keeps the SDK's own message, capped at 120
characters (the SDK masks credentials in it); an unknown choice label is quoted,
capped at 40 characters; an out-of-range score or probability is quoted as a
number. No other part of a response body is used, so a provider echoing state
back can leak at most a 40-character label into a log.

## Who can call it

Switchboard binds to `127.0.0.1` and has no authentication, which is right for
one user on one machine. The remaining gap on such a service is the browser: a
web page open on this machine can send requests to localhost. So
`security.LocalBoundary` sits in front of every route, `/mcp` included:

* a request carrying an `Origin` header must come from Switchboard's own
  origin (the dashboard), or it gets **403**. Native clients - cptr, SDKs,
  curl, MCP clients - send no `Origin`, so they are unaffected;
* a request with a body must be `application/json` (**415** otherwise), which a
  browser cannot send cross-origin without a preflight;
* request bodies are capped at 8 MB (**413**), counted in bytes so a chunked
  upload cannot get around it.

This is protection for a local service, not authentication. Exposing
Switchboard beyond loopback would need real credentials.

## What Switchboard reads but does not touch

* `C:\AI\Config\models.json` - read for model parameters. Written **only** by
  the dashboard, when a switch happens, and backed up first.
* `C:\AI\models\*.gguf` - headers read, never modified, moved or deleted.
* `C:\AI\Projects\lecture-notes\state\status.json` - read (never written) for
  the recording guard: only `active`, `class_code`, `started_at` and
  `updated_at` are used. The class code appears in routing reasons and on the
  dashboard; nothing from the transcript is read.
* `~/.cptr/config.toml` - read by `doctor` to report integration status, and
  written once by `integrate-cptr.ps1` with a backup.

The cptr config file stores a Browser Use API key in plaintext. Switchboard
never transmits it: `doctor` reads the file only to check that Switchboard's
own URL is in it, and the redaction rules would strip a `bu_…` token from
anything outbound. But `integrate-cptr.ps1` copies the whole file, key
included, to `backups\cptr-config.toml.<timestamp>.bak` (git-ignored) before it
edits the config, so such a backup holds the same plaintext key. The one taken
on 2026-09-21 was moved to the Recycle Bin on 2026-09-23; running the script
again would take a new one.

## The upstream's own exposure

`llama-server` is launched by `C:\AI\start.ps1` with `--host 0.0.0.0` and no
API key, so it is reachable from the local network, and llama.cpp logs that
warning itself at startup. Switchboard binds to `127.0.0.1` only and does not
change how llama-server is launched. Narrowing that is a change to
`Build-LlamaArgs.ps1`, outside this project's remit, but worth knowing about.
