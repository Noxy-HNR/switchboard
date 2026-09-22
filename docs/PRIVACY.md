# Privacy

Two questions, answered separately: what leaves the machine, and what is kept
on it.

## What leaves the machine

**Only the Jev backend sends anything anywhere**, and only when
`TYPESAFE_API_KEY` is set and `privacy.allow_cloud` is true. With no key,
Switchboard is entirely local: the logit and rule backends never open an
outbound connection, and the local adapter talks only to `127.0.0.1:10000`.

When Jev is enabled, one request per decision carries a small JSON object:

```json
{"request": {"message": "<latest user turn>",
             "previous_message": "<the one before it>",
             "available_tools": ["read_file", "run_command"],
             "turn": 3, "has_images": false}}
```

Before any of that is assembled, every string goes through three steps:

1. **Pasted machine output stays on this machine.** Python tracebacks, fenced
   code blocks, runs of terminal output (shell prompts, log lines, stack frames,
   `KEY=value` environment lines), JSON and XML/HTML documents, and PEM key
   material are replaced by a one-line description:

   ```
   Why does this fail?
   [Python traceback kept local: 23 lines]
   ```

   The contents never leave; the user's own words around them do, because
   those are what the routing judgments are actually about. The detectors err
   toward removing too much, and a pasted JSON blob is recognised by its shape
   even when it is truncated and would not parse. This step cannot be turned
   off, and it does not depend on redaction - it is checked with redaction
   disabled.
2. **Secrets are stripped** from what remains. `privacy.py` replaces
   credential-shaped spans with typed placeholders: API keys (OpenAI,
   Anthropic, GitHub, AWS, Slack, Google, Browser Use, TypeSafe), bearer and
   basic tokens, `Authorization:` and `Cookie:` headers, JWTs, private key
   blocks, connection strings with passwords, `key = value` and JSON
   `"key": "value"` pairs whose name looks like a credential, `.env`-shaped
   lines, email addresses and card-shaped digit runs.
3. **One shared budget.** `privacy.max_state_chars` (1500 by default) caps the
   whole state, not each field: the current message is spent first, then the
   previous one, then tool names. Long text keeps both ends, because the topic
   is at the start and the question at the end.

Which *kinds* of content were withheld is recorded in the decision's notes -
"kept local: Python traceback", "redacted: openai-key, email" - and the content
itself is recorded nowhere.

Only the latest turn goes: not the conversation, not file contents, not tool
*results*, only tool **names**. A 40-message conversation contributes its last
two user turns and a count.

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

The SQLite database at `data/switchboard.sqlite3`. Its rules are enforced in
code, not by convention:

* **No column can hold a prompt.** Requests are identified by a SHA-256 prefix.
  The store rejects a "hash" longer than 64 characters or containing a newline,
  so a caller cannot smuggle text through the hash column either.
* **No previews by default.** `privacy.store_previews` is **off**: the
  database holds a hash of each request and nothing of its text. Turning it on
  keeps an 80-character excerpt for debugging, which the router redacts and
  the store then redacts and truncates again regardless of what it was given;
  `privacy.preview_chars` can only tighten that limit, never raise it past 200.
* **Agent steps arrive hashed.** Goals and actions are hashed by the router
  before the store sees them.
* **Nothing upstream says is kept.** When llama-server or a decision provider
  answers with an error, the database records the status or the exception
  type - never the body, which can echo the request back.

To keep a short redacted excerpt for debugging:

```toml
[privacy]
store_previews = true
```

### What the logs hold

`logs/switchboard.out.log` and `.err.log` carry uvicorn's startup lines and
warnings. Request bodies are not logged. The Jev backend builds its error
messages from the exception type and status code only, never the response body,
so a provider echoing state back cannot leak it into a log.

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
* `~/.cptr/config.toml` - read by `doctor` to report integration status, and
  written once by `integrate-cptr.ps1` with a backup.

The cptr config file stores a Browser Use API key in plaintext. Switchboard
does not read, copy or transmit it; its redaction rules would strip a `bu_…`
token from anything outbound. Flagged here because it is worth knowing.

## The upstream's own exposure

`llama-server` is launched by `C:\AI\start.ps1` with `--host 0.0.0.0` and no
API key, so it is reachable from the local network, and llama.cpp logs that
warning itself at startup. Switchboard binds to `127.0.0.1` only and does not
change how llama-server is launched. Narrowing that is a change to
`Build-LlamaArgs.ps1`, outside this project's remit, but worth knowing about.
