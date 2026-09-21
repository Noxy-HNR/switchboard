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
    ("auth-header", re.compile(r"(?i)\b(authorization|proxy-authorization)\s*:\s*\S+")),
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
            r"|client[_-]?secret|private[_-]?key)[a-z0-9_.-]*)\b\s*[:=]\s*[\"']?([^\s\"',;]{6,})"
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
    if limit <= 0 or len(text) <= limit:
        return text
    if limit <= 32:
        return text[:limit]
    head = (limit * 2) // 3
    tail = limit - head - 15
    return f"{text[:head]}\n[... {len(text) - head - tail} chars omitted ...]\n{text[-tail:]}"


def prepare_cloud_text(text: str, *, max_chars: int, do_redact: bool = True) -> tuple[str, list[str]]:
    """Redact then shorten a piece of text destined for a cloud provider."""
    kinds: list[str] = []
    prepared = text
    if do_redact:
        result = redact(text)
        prepared, kinds = result.text, result.kinds
    return truncate_middle(prepared, max_chars), kinds
