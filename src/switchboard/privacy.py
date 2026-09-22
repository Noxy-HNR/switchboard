"""Redaction, hashing, and previews.

Two rules drive this module:

* Nothing leaves the machine that was not reduced to the smallest state a
  decision needs, with credential-shaped text removed first.
* Nothing durable (database rows, logs) holds a raw prompt. Requests are
  identified by hash, with an optional short redacted preview for debugging.

Redaction is pattern based, so it is a safety net rather than a guarantee. It is
applied to cloud state and to stored previews alike.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field

_PLACEHOLDER = "[REDACTED:{kind}]"

# Ordered: the first pattern that matches a span owns it. Specific shapes come
# before generic key=value, so a match reports the most informative kind.
_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "private-key",
        re.compile(
            r"-----BEGIN[ A-Z]*PRIVATE KEY-----.*?-----END[ A-Z]*PRIVATE KEY-----",
            re.DOTALL,
        ),
    ),
    ("auth-header", re.compile(r'''(?im)\b(?:authorization|proxy-authorization|cookie|set-cookie)["']?\s*:\s*[^\r\n]+''')),
    ("bearer", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-~+/]{12,}=*")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}")),
    ("openai-key", re.compile(r"\bsk-[A-Za-z0-9_-]{16,}")),
    ("anthropic-key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{16,}")),
    ("github-token", re.compile(r"\b(gh[pousr]_[A-Za-z0-9]{16,}|github_pat_[A-Za-z0-9_]{20,})")),
    ("aws-key", re.compile(r"\b(AKIA|ASIA)[A-Z0-9]{16}\b")),
    ("slack-token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}")),
    ("google-key", re.compile(r"\bAIza[A-Za-z0-9_-]{20,}")),
    ("browser-use-key", re.compile(r"\bbu_[A-Za-z0-9_-]{16,}")),
    ("typesafe-key", re.compile(r"\bts-[A-Za-z0-9_-]{16,}")),
    (
        "connection-string",
        re.compile(r"(?i)\b[a-z][a-z0-9+.-]*://[^\s:/@]+:[^\s/@]+@\S+"),
    ),
    (
        # KEY=value / "api_key": "value" / password: value
        "credential",
        re.compile(
            r"(?i)\b([a-z0-9_.-]*(?:api[_-]?key|secret|passwd|password|token|access[_-]?key"
            r"|client[_-]?secret|private[_-]?key)[a-z0-9_.-]*)\b[\"']?\s*[:=]\s*(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)"
        ),
    ),
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    # 13-19 digits, optionally grouped: card-shaped. Plain years/ports are shorter.
    ("long-number", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
)

# Matched as whole lines, because an .env dump has no other structure to key on.
_ENV_LINE = re.compile(
    r"(?im)^\s*(?:export\s+)?([A-Z][A-Z0-9_]{2,})\s*=\s*(\S.*)$",
)
_ENV_SENSITIVE = re.compile(
    r"(?i)(key|secret|token|password|passwd|credential|auth|private|session|cookie)"
)


@dataclass
class RedactionResult:
    text: str
    kinds: list[str] = field(default_factory=list)

    @property
    def redacted(self) -> bool:
        return bool(self.kinds)


def redact(text: str) -> RedactionResult:
    """Replace credential-shaped spans with typed placeholders."""
    if not text:
        return RedactionResult(text="", kinds=[])

    kinds: list[str] = []

    def _env_line(match: re.Match[str]) -> str:
        name, value = match.group(1), match.group(2)
        if not _ENV_SENSITIVE.search(name) or len(value) < 4:
            return match.group(0)
        kinds.append("env-var")
        return f"{name}={_PLACEHOLDER.format(kind='env-var')}"

    result = _ENV_LINE.sub(_env_line, text)

    for kind, pattern in _PATTERNS:

        def _sub(match: re.Match[str], kind: str = kind) -> str:
            kinds.append(kind)
            if kind == "credential":
                return f"{match.group(1)}={_PLACEHOLDER.format(kind=kind)}"
            return _PLACEHOLDER.format(kind=kind)

        result = pattern.sub(_sub, result)

    # Preserve first-seen order without duplicates.
    return RedactionResult(text=result, kinds=list(dict.fromkeys(kinds)))


def contains_secret(text: str) -> bool:
    """Whether redaction would change this text."""
    return redact(text).redacted


def hash_text(text: str, *, length: int = 16) -> str:
    """Stable identifier for a prompt, safe to store."""
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:length]


def preview(text: str, *, limit: int = 80, redact_first: bool = True) -> str:
    """A short, redacted, single-line excerpt for debugging."""
    source = redact(text).text if redact_first else text
    collapsed = " ".join(source.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1].rstrip() + "\u2026"


def truncate_middle(text: str, limit: int) -> str:
    """Trim to `limit` characters, keeping both ends.

    The beginning of a request carries the topic and the end carries the actual
    question, so a middle ellipsis preserves more signal than a head-only cut.
    """
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    if limit <= 32:
        return text[:limit]
    marker = "\n[... omitted ...]\n"
    available = limit - len(marker)
    head = available * 2 // 3
    tail = available - head
    return f"{text[:head]}{marker}{text[-tail:]}"


# --- keeping pasted machine output local -------------------------------------
#
# The rule: raw terminal output, file contents and structured dumps never go to
# a cloud provider. The request *around* them still may, because judging what a
# request is about needs the words the user wrote, not the bytes they pasted.
# "Why does this fail? <40-line traceback>" becomes "Why does this fail?
# [Python traceback kept local: 40 lines]", which is both private and a better
# state: the TypeSafe docs warn that irrelevant detail in the state costs
# accuracy, and a traceback's *presence* is the signal, not its contents.
#
# Every detector errs toward removing too much. A false positive costs a little
# context; a false negative sends someone's terminal to the cloud.

_FENCE = re.compile(r"```[ \t]*([\w+#.-]*)[^\n]*\n?(.*?)(?:```|\Z)", re.DOTALL)
_PEM_BLOCK = re.compile(r"-----BEGIN [A-Z0-9 ]+-----.*?(?:-----END [A-Z0-9 ]+-----|\Z)", re.DOTALL)
_TRACEBACK_START = re.compile(r"^\s*Traceback \(most recent call last\):")
_EXCEPTION_LINE = re.compile(r"^[A-Za-z_][\w.]*(?:Error|Exception|Warning|Exit|Interrupt)\b")
# Lines that are machine output on their own: shell prompts, stack frames from
# several languages, timestamped log lines, environment assignments, and the
# `file:line:` shape compilers and linters print.
_MACHINE_LINE = re.compile(
    r"""^\s*(?:
        PS\ [A-Za-z]:\\[^>]*>          # PowerShell prompt
      | [A-Za-z]:\\[^>\n]*>             # cmd prompt
      | \$\ \S                          # POSIX prompt
      | >>>\ |\.\.\.\                   # Python REPL
      | File\ "[^"]+",\ line\ \d+       # Python frame
      | at\ [\w.$<>]+\(.*\)             # Java / .NET / JS frame
      | \d{4}-\d\d-\d\d[T\ ]\d\d:\d\d  # ISO timestamp log line
      | \[?\d\d:\d\d:\d\d[.,\]]         # clock timestamp log line
      | (?:export\ )?[A-Z][A-Z0-9_]{2,}\s*=  # environment assignment
      | [\w./\\-]+\.\w+:\d+(?::\d+)?:\ # file:line: diagnostics
      | (?:INFO|WARN|WARNING|ERROR|DEBUG|TRACE|FATAL)\b[:\ ]
    )""",
    re.VERBOSE,
)
# A run of machine lines this long is output, not a sentence that mentions a path.
_MACHINE_RUN_MIN = 3


def _placeholder(kind: str, text: str) -> str:
    lines = text.count("\n") + 1 if text.strip() else 0
    unit = "line" if lines == 1 else "lines"
    return f"[{kind} kept local: {lines} {unit}]"


def _structured_document(text: str) -> str | None:
    """A whole message that is a JSON or XML document, rather than a question."""
    stripped = text.strip()
    if stripped.startswith(("<?xml", "<!DOCTYPE", "<html")):
        return "XML/HTML document"
    if stripped[:1] in "{[" and stripped[-1:] in "}]":
        try:
            json.loads(stripped)
        except ValueError:
            return None
        return "JSON document"
    return None


def minimize_for_cloud(text: str) -> tuple[str, list[str]]:
    """Replace pasted machine output with a short description of it.

    Returns the minimized text and the kinds of content that were kept local,
    in first-seen order. Prose is returned unchanged: redaction and truncation
    are separate steps (see `prepare_cloud_text`).
    """
    kept: list[str] = []

    def note(kind: str) -> None:
        if kind not in kept:
            kept.append(kind)

    document = _structured_document(text)
    if document:
        note(document)
        return _placeholder(document, text.strip()), kept

    def fence(match: re.Match[str]) -> str:
        language = match.group(1).strip()
        kind = f"{language} code block" if language else "code block"
        note("code block")
        return _placeholder(kind, match.group(2).rstrip("\n"))

    result = _FENCE.sub(fence, text)

    def pem(match: re.Match[str]) -> str:
        note("key material")
        return "[key material kept local]"

    result = _PEM_BLOCK.sub(pem, result)

    lines = result.split("\n")
    output: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]

        if _TRACEBACK_START.match(line):
            end = index + 1
            # A Python traceback: indented frames, then one unindented exception line.
            while end < len(lines) and lines[end][:1] in {" ", "\t"}:
                end += 1
            if end < len(lines) and _EXCEPTION_LINE.match(lines[end]):
                end += 1
            note("Python traceback")
            output.append(_placeholder("Python traceback", "\n".join(lines[index:end])))
            index = end
            continue

        stripped = line.strip()
        if stripped[:1] in {"{", "["} or (stripped.startswith("<") and ">" in stripped):
            # A pasted document inside a message: JSON, a log object, XML/HTML.
            # Take the whole paragraph. Pasted JSON is often truncated and will
            # not parse, so structure is judged by shape, not by json.loads.
            end = index
            while end < len(lines) and lines[end].strip():
                end += 1
            block = "\n".join(lines[index:end])
            if any(mark in block for mark in ('"', ":", "=", "</", "/>")):
                kind = "markup" if stripped.startswith("<") else "JSON document"
                note(kind)
                output.append(_placeholder(kind, block))
                index = end
                continue

        if _MACHINE_LINE.match(line):
            end = index
            while end < len(lines) and (
                _MACHINE_LINE.match(lines[end])
                or _EXCEPTION_LINE.match(lines[end])
                or (lines[end][:1] in {" ", "\t"} and lines[end].strip())
            ):
                end += 1
            if end - index >= _MACHINE_RUN_MIN:
                note("terminal output")
                output.append(_placeholder("terminal output", "\n".join(lines[index:end])))
                index = end
                continue

        output.append(line)
        index += 1

    return "\n".join(output), kept


def prepare_cloud_text(text: str, *, max_chars: int, do_redact: bool = True) -> tuple[str, list[str]]:
    """Redact then shorten a piece of text destined for a cloud provider.

    Machine output is not handled here: callers run `minimize_for_cloud` first,
    so its placeholders survive and its contents never reach redaction at all.
    """
    kinds: list[str] = []
    prepared = text
    if do_redact:
        result = redact(text)
        prepared, kinds = result.text, result.kinds
    return truncate_middle(prepared, max_chars), kinds
