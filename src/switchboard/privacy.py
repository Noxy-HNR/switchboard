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
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import NamedTuple

_PLACEHOLDER = "[REDACTED:{kind}]"

# Ordered: the first pattern that matches a span owns it. Specific shapes come
# before generic key=value, so a match reports the most informative kind.
#
# Every pattern runs over whatever a client sent, so each is linear in the text:
# a pattern that could fail late (after scanning a long run) may only start at
# the beginning of that run, which the lookbehinds below enforce, and runs that
# are followed by a fixed delimiter are matched possessively. A quadratic
# pattern here would let one crafted request stall the event loop.
_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "private-key",
        re.compile(
            # An unterminated block runs to the end of the text: it is still a
            # key, and a lazy scan with no end would retry from every BEGIN.
            r"-----BEGIN[ A-Z]*PRIVATE KEY-----.*?(?:-----END[ A-Z]*PRIVATE KEY-----|\Z)",
            re.DOTALL,
        ),
    ),
    ("auth-header", re.compile(r'''(?im)\b(?:authorization|proxy-authorization|cookie|set-cookie)["']?\s*:\s*[^\r\n]+''')),
    ("bearer", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-~+/]{12,}=*")),
    # From the start of a base64url run: "-" is in the run, so a word boundary
    # alone let every "eyJ" in "eyJ-eyJ-..." rescan the rest of it.
    ("jwt", re.compile(r"(?<![A-Za-z0-9_-])eyJ[A-Za-z0-9_-]{8,}+\.[A-Za-z0-9_-]{8,}+\.[A-Za-z0-9_-]{8,}")),
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
        # A scheme is short; bounding it keeps a long dotted word from being
        # rescanned from every letter in it.
        re.compile(r"(?i)\b[a-z][a-z0-9+.-]{0,40}+://[^\s:/@]++:[^\s/@]++@\S+"),
    ),
    (
        # KEY=value / "api_key": "value" / password: value
        "credential",
        # The name is the whole run of name characters, taken from its start,
        # and must contain a credential word somewhere in it.
        re.compile(
            r"(?i)(?<![a-z0-9_.-])"
            r"(?=[a-z0-9_.-]*?(?:api[_-]?key|secret|passwd|password|token|access[_-]?key"
            r"|client[_-]?secret|private[_-]?key"
            # PASS and PWD only as a whole segment, so "bypass" and "passport" are not names.
            r"|(?<![a-z0-9])(?:pass|pwd)(?![a-z0-9])))"
            r"([a-z0-9_.-]++)[\"']?\s*[:=]\s*(\"[^\"]*\"|'[^']*'|[^\s,;]+)"
        ),
    ),
    # From the start of the run, so "a.b.c.d..." with no "@" is scanned once.
    ("email", re.compile(r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]++@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    # US social security number, 3-2-4 with dashes. Nine digits is too short for
    # long-number, and the 3-2-4 grouping cannot match inside an ISO date (4-2-2)
    # or a 4-4-4-4 card, so neither of those changes kind.
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    # 13-19 digits, optionally grouped: card-shaped. Plain years/ports are shorter.
    ("long-number", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
)

# Matched as whole lines, because an .env dump has no other structure to key on.
_ENV_LINE = re.compile(
    # Horizontal whitespace only: "\s" would reach across blank lines, making
    # every line start rescan all the blank lines below it.
    r"(?im)^[ \t]*+(?:export[ \t]++)?([A-Z][A-Z0-9_]{2,}+)[ \t]*+=[ \t]*+(\S.*)$",
)
_ENV_SENSITIVE = re.compile(
    r"(?i)(key|secret|token|password|passwd|credential|auth|private|session|cookie"
    r"|(?<![a-z0-9])(?:pass|pwd)(?![a-z0-9]))"
)


_STRONG_CREDENTIAL_WORDS = ("key", "secret", "pass", "pwd", "private", "auth")
# Up to nine digits, optionally quoted, then only closing punctuation.
_COUNT_VALUE = re.compile(r"[\"']?\d{1,9}[\"']?[)\]}.!?]*")


def _is_token_count(name: str, value: str) -> bool:
    """`max_tokens: 256` and `"prompt_tokens": 21` are counts, not credentials.

    Only a count-shaped name qualifies - the plural "tokens" - and only with a
    short all-digit value. A singular token name is a credential whatever its
    length: one-time, reset, session and SMS codes are four to eight digits. A
    name that also says key, secret, pass, pwd, private or auth keeps every value
    redacted.
    """
    lowered = name.lower()
    if "tokens" not in lowered or any(word in lowered for word in _STRONG_CREDENTIAL_WORDS):
        return False
    # The value runs to the next space, comma or semicolon, so it can carry the
    # closer of the JSON or sentence around it: `37}` or `160).`.
    return _COUNT_VALUE.fullmatch(value) is not None


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
            if kind == "credential" and _is_token_count(match.group(1), match.group(2)):
                return match.group(0)
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
#
# Each removed block becomes one line: "[pasted {kind} omitted: N lines]" ("1
# line" when singular), or "[pasted key material omitted]" for a PEM block. The
# decision model reads that literally, so it says only that the user pasted
# something the model cannot see. The earlier "kept local" wording read as local
# files and as missing evidence, which pushed "needs tools" judgments up. Code
# and question text downstream match "[pasted ... omitted", so the format is
# fixed.
#
# "Why does this fail? <40-line traceback>" becomes "Why does this fail?" and
# "[pasted Python traceback omitted: 40 lines]", which is both private and a
# better state: the TypeSafe docs warn that irrelevant detail in the state costs
# accuracy, and a traceback's *presence* is the signal, not its contents.
#
# Detection runs in layers, most specific first:
#
# 1. Fenced code blocks and PEM blocks, found by their delimiters.
# 2. Line by line: Python tracebacks; shell prompts (a prompt means what follows
#    is output, so one prompt line starts a run that lasts to the next blank
#    line); runs of machine-shaped lines such as stack frames, log lines,
#    compiler diagnostics and directory listings; JSON and markup documents.
# 3. A shape backstop for what no signature knows, such as unfenced YAML, CSV
#    or Python: a paragraph of 4+ lines where fewer than half end like a
#    sentence and at least half carry data marks (":", "=", "|", "{", ";", a
#    tab, indentation) becomes "[pasted text omitted: N lines]".
#
# A placeholder line passes through every later layer unchanged. Placeholders
# contain "[" and ":", so without that rule one was re-read as a JSON document
# and took the question under it along. Sentence-shaped lines at the edges of a
# removed JSON block or backstop paragraph are kept for the same reason: with
# no blank line, that is where the question sits.
#
# Every detector errs toward removing too much. A false positive costs a little
# context; a false negative sends someone's terminal to the cloud.

# The info string after the opening fence stops at a closing fence, so a
# one-line ```span``` ends where it closes instead of taking the rest of the text.
_FENCE = re.compile(r"```[ \t]*([\w+#.-]*)(?:(?!```)[^\n])*\n?(.*?)(?:```|\Z)", re.DOTALL)
# A tilde fence opens only at the start of a line (CommonMark allows up to three
# spaces of indent) and closes on a tilde line at least as long. Anchoring
# matters: PowerShell underlines an error with "+ ~~~~", which is not a fence.
_TILDE_FENCE = re.compile(
    r"^[ ]{0,3}(~{3,})[ \t]*([\w+#.-]*)[^\n]*\n(.*?)(?:^[ ]{0,3}\1~*[ \t]*$|\Z)",
    re.DOTALL | re.MULTILINE,
)
# Fence info strings that may appear in a placeholder: the kind of code is a
# useful routing signal, and a name on this list cannot carry anything private.
_FENCE_LANGUAGES = frozenset(
    {
        "bash", "bat", "c", "c#", "c++", "cmd", "console", "cpp", "cs", "csharp",
        "css", "csv", "diff", "dockerfile", "go", "html", "ini", "java",
        "javascript", "js", "json", "jsx", "kotlin", "log", "lua", "makefile",
        "markdown", "md", "php", "powershell", "ps1", "pwsh", "py", "python",
        "r", "rb", "ruby", "rust", "rs", "scala", "sh", "shell", "sql", "swift",
        "text", "toml", "ts", "tsx", "txt", "typescript", "xml", "yaml", "yml",
        "zsh",
    }
)
_PEM_BLOCK = re.compile(r"-----BEGIN [A-Z0-9 ]+-----.*?(?:-----END [A-Z0-9 ]+-----|\Z)", re.DOTALL)
_TRACEBACK_START = re.compile(r"^\s*+Traceback \(most recent call last\):")
_EXCEPTION_LINE = re.compile(r"^[A-Za-z_][\w.]*(?:Error|Exception|Warning|Exit|Interrupt)\b")
# A placeholder this module wrote, as a whole line or inside one.
# The kind is bounded, so "[pasted [pasted ..." without a close is not rescanned
# from every opening.
_PLACEHOLDER_TEXT = re.compile(r"\[pasted [^\]\n]{1,60}? omitted(?:: \d+ lines?)?\]")
# A shell prompt. The character before ">" may not be a space, "-" or "=", so a
# sentence that starts with a path and later says "->" or ">=" is not a prompt;
# "$ " may not be followed by a digit, so "$ 5 is the fee" is not one either.
_PROMPT_LINE = re.compile(
    r"""^\s*+(?:\([\w.-]+\)\ )?                 # optional (venv) prefix
    (?:
        PS\ [A-Za-z]:\\(?:[^>\n]*[^\s>=-])?>   # PowerShell: PS C:\AI>
      | [A-Za-z]:\\(?:[^>\n]*[^\s>=-])?>       # cmd: C:\AI>
      | \$\ [^\s\d]                            # POSIX: $ ls
      | [\w.-]+@[\w.-]+:[^\s$#]*[$#]\ \S       # POSIX: sam@box:~/app$ ls
      | [\w.-]+@[\w.-]+\ [^\s%#]+\ [%#]\ \S       # zsh: sam@box ~ % ls
    )""",
    re.VERBOSE,
)
# Lines that look like machine output: stack frames from several languages,
# timestamped log lines, environment assignments, compiler and linter
# diagnostics, PowerShell error records, npm, directory listings, git status.
# Any one of these can also open a sentence, so it takes a run of them
# (_MACHINE_RUN_MIN) to count as output.
_MACHINE_LINE = re.compile(
    r"""^\s*+(?:
        >>>\ |\.\.\.\                     # Python REPL
      | File\ "[^"]+",\ line\ \d+         # Python frame
      | at\ [\w.$<>]+\(.*\)               # Java / .NET / JS frame
      | \[?\d{4}-\d\d-\d\d[T\ ]\d\d:\d\d  # ISO timestamp log line
      | \[?\d\d:\d\d:\d\d[.,\]]           # clock timestamp log line
      | (?:export\ )?[A-Z][A-Z0-9_]{2,}\s*=  # environment assignment
      | [\w./\\-]+\.\w+:\d+(?::\d+)?:\    # file:line: diagnostics
      | (?:INFO|WARN|WARNING|ERROR|DEBUG|TRACE|FATAL)\b[:\ ]
      | (?:[Ee]rror|[Ww]arning)(?:\[\w+\])?:\   # error: / warning[W0612]:
      | -->\                              # rustc source pointer
      | \d*+\s*+\|                        # rustc gutter ("5 |", "  |"), table row
      | [A-Z][a-z]+-[A-Z]\w*\ :\          # PowerShell error: "Remove-Item : ..."
      | npm\ (?:ERR!|WARN)                # npm
      | \d\d/\d\d/\d{4}\                  # cmd dir entry
      | total\ \d+\s*$                    # ls -l header
      | On\ branch\                       # git status
      | Your\ branch\                     # git status
      | Changes\ (?:not\ staged|to\ be\ committed)  # git status
      | Untracked\ files:                 # git status
    )""",
    re.VERBOSE,
)
# Lines nobody types as prose. One of these makes its run output however short
# the run is: a two-line `npm ERR!` block still names paths.
_OUTPUT_MARKER = re.compile(
    r"""^\s*+(?:
        npm\ ERR!                                  # npm failure
      | At\ line:\d+\ char:\d+                     # PowerShell error position
      | (?:error|warning)\[\w+\]:                  # rustc: error[E0382]:
      | -->\ \S+:\d+                               # rustc: --> src/main.rs:5:20
      | [-dlcbps][rwxsStT-]{9}[.+@]?\ +\d          # ls -l entry
      | \d\d/\d\d/\d{4}\ +\d\d:\d\d(?:\ [AP]M)?\ +(?:<[A-Z]+>|[\d,.]+\ )  # cmd dir entry
      | Volume\ in\ drive\ [A-Za-z]\               # cmd dir header
      | Volume\ Serial\ Number\ is\                # cmd dir header
      | Directory\ of\ [A-Za-z]:\\                 # cmd dir header
      | Changes\ (?:not\ staged\ for\ commit|to\ be\ committed):  # git status
      | Exception\ in\ thread\ "                   # Java uncaught exception
      | Unhandled\ exception\.                     # .NET uncaught exception
    )""",
    re.VERBOSE,
)
# A run of machine lines this long is output, not a sentence that mentions a path.
_MACHINE_RUN_MIN = 3
# A JSON document starts with "{", or with "[" followed by something a JSON
# array holds. "[Question]: ..." and "[x] done" are prose and must not match.
_DOCUMENT_START = re.compile(r'\s*+(?:\{|\[\s*+(?:["{\[]|\d|$))')
# The backstop. Four lines, because three short lines are too often a list the
# user typed ("1. Install it" ...).
_SHAPE_MIN_LINES = 4
_STRUCTURE_MARK = re.compile(r"[:=|{;\t]")
_SENTENCE_END = (".", "?", "!")


class _Block(NamedTuple):
    """Machine output found by a line detector: it runs up to `end` (exclusive)."""

    kind: str
    end: int


def _placeholder(label: str, text: str) -> str:
    lines = text.count("\n") + 1 if text.strip() else 0
    unit = "line" if lines == 1 else "lines"
    return f"[pasted {label} omitted: {lines} {unit}]"


def _is_placeholder_line(line: str) -> bool:
    return _PLACEHOLDER_TEXT.fullmatch(line.strip()) is not None


def _is_machine_line(line: str) -> bool:
    return bool(_MACHINE_LINE.match(line) or _OUTPUT_MARKER.match(line))


def _continues_run(line: str) -> bool:
    """A line that extends a run of output but is too ambiguous to start one.

    Indented detail and exception lines belong to the output above them.
    PowerShell's "+ " error-record lines are here rather than in _MACHINE_LINE
    because a "+" bulleted list the user typed looks the same.
    """
    if not line.strip():
        return False
    return line[:1] in {" ", "\t"} or line.startswith("+ ") or bool(_EXCEPTION_LINE.match(line))


def _paragraph_end(lines: list[str], start: int) -> int:
    """Index one past the paragraph at `start`. A placeholder line ends it too."""
    end = start
    while end < len(lines) and lines[end].strip() and not _is_placeholder_line(lines[end]):
        end += 1
    return end


def _ends_sentence(line: str) -> bool:
    return _PLACEHOLDER_TEXT.sub("", line).rstrip().endswith(_SENTENCE_END)


def _has_structure(line: str) -> bool:
    # A placeholder inside a line has a ":" of its own that says nothing about
    # the user's text.
    return line[:1] in {" ", "\t"} or _STRUCTURE_MARK.search(_PLACEHOLDER_TEXT.sub("", line)) is not None


def _reads_as_sentence(line: str) -> bool:
    """A line a person wrote: starts with a word, ends like a sentence, no data marks.

    Only such lines are kept at the edge of a removed block, so the test is
    strict: "Why does this fail?" passes; "retries: 3." and '"done."' do not.
    """
    return line[:1].isalnum() and _ends_sentence(line) and not _has_structure(line)


_DOCUMENT_MARKS = ('"', ":", "=", "</", "/>")


class _Lines:
    """A message's lines plus the paragraph facts detectors look up.

    A detector runs at every line, and some need facts about the rest of the
    paragraph: where it ends, whether any line in it carries a data mark, where
    its trailing sentences begin. Recomputing those from each line made a long
    paragraph quadratic, so they are computed once here, back to front.
    """

    def __init__(self, lines: list[str]) -> None:
        count = len(lines)
        self.lines = lines
        # paragraph_end[i]: one past the paragraph at i (see _paragraph_end).
        self.paragraph_end = [count] * (count + 1)
        # next_mark[i]: the first line at or after i with a document mark.
        self.next_mark = [count] * (count + 1)
        for index in range(count - 1, -1, -1):
            line = lines[index]
            breaks = not line.strip() or _is_placeholder_line(line)
            self.paragraph_end[index] = index if breaks else self.paragraph_end[index + 1]
            has_mark = any(mark in line for mark in _DOCUMENT_MARKS)
            self.next_mark[index] = index if has_mark else self.next_mark[index + 1]
        # sentence_run[e]: where the run of sentence lines ending just before e starts.
        self.sentence_run = [0] * (count + 1)
        for end in range(1, count + 1):
            reads = _reads_as_sentence(lines[end - 1])
            self.sentence_run[end] = self.sentence_run[end - 1] if reads else end

    def __len__(self) -> int:
        return len(self.lines)

    def __getitem__(self, index: int) -> str:
        return self.lines[index]


# --- line detectors: each returns the block starting at lines[index], or None


def _traceback_at(lines: _Lines, index: int) -> _Block | None:
    if not _TRACEBACK_START.match(lines[index]):
        return None
    # A Python traceback: indented frames, then one unindented exception line.
    end = index + 1
    while end < len(lines) and lines[end][:1] in {" ", "\t"}:
        end += 1
    if end < len(lines) and _EXCEPTION_LINE.match(lines[end]):
        end += 1
    return _Block("Python traceback", end)


def _prompt_run_at(lines: _Lines, index: int) -> _Block | None:
    # What follows a prompt is that command's output, whatever it looks like,
    # so the run takes the rest of the paragraph.
    if not _PROMPT_LINE.match(lines[index]):
        return None
    return _Block("terminal output", lines.paragraph_end[index])


def _machine_run_at(lines: _Lines, index: int) -> _Block | None:
    # An exception line may open a run too: a JS or Java trace puts its
    # "TypeError: ..." line above the frames, not below them.
    if not (_is_machine_line(lines[index]) or _EXCEPTION_LINE.match(lines[index])):
        return None
    end = index
    prompted = False
    while end < len(lines) and not _is_placeholder_line(lines[end]):
        line = lines[end]
        if _PROMPT_LINE.match(line):
            end, prompted = lines.paragraph_end[end], True
            break
        if not (_is_machine_line(line) or _continues_run(line)):
            break
        end += 1
    run = lines.lines[index:end]
    # One machine-looking line is usually a sentence that mentions a path or a
    # variable. A run of them, a prompt, or an unmistakable marker is output.
    if len(run) >= _MACHINE_RUN_MIN or prompted or any(_OUTPUT_MARKER.match(line) for line in run):
        return _Block("terminal output", end)
    return None


def _document_at(lines: _Lines, index: int) -> _Block | None:
    stripped = lines[index].strip()
    markup = stripped.startswith("<") and ">" in stripped
    if not markup and not _DOCUMENT_START.match(lines[index]):
        return None
    # A pasted document inside a message: JSON, a log object, XML/HTML. Take
    # the whole paragraph. Pasted JSON is often truncated and will not parse,
    # so structure is judged by shape, not by json.loads.
    end = lines.paragraph_end[index]
    if not markup:
        # JSON cannot hold a bare sentence, so sentence lines at the end are
        # the question typed under the paste. Markup can, so it keeps them.
        end = max(index + 1, lines.sentence_run[end])
    if lines.next_mark[index] >= end:
        return None
    return _Block("markup" if markup else "JSON document", end)


def _block_at(lines: _Lines, index: int) -> _Block | None:
    if _is_placeholder_line(lines[index]):
        return None
    return (
        _traceback_at(lines, index)
        or _prompt_run_at(lines, index)
        or _machine_run_at(lines, index)
        or _document_at(lines, index)
    )


def _replace_blocks(lines: list[str], note: Callable[[str], None]) -> list[str]:
    """Replace each block of machine output with its placeholder line."""
    indexed = _Lines(lines)
    output: list[str] = []
    index = 0
    while index < len(lines):
        block = _block_at(indexed, index)
        if block is None:
            output.append(lines[index])
            index += 1
            continue
        note(block.kind)
        output.append(_placeholder(block.kind, "\n".join(lines[index : block.end])))
        index = block.end
    return output


# --- shape backstop -----------------------------------------------------------


def _reads_as_data(paragraph: list[str], core: list[str]) -> bool:
    """Whether a paragraph is shaped like data rather than sentences."""
    if len(paragraph) < _SHAPE_MIN_LINES:
        return False
    if sum(1 for line in paragraph if _ends_sentence(line)) * 2 >= len(paragraph):
        return False
    if sum(1 for line in paragraph if _has_structure(line)) * 2 >= len(paragraph):
        return True
    # Comma-separated rows carry none of the marks above. Prose lines almost
    # never share one exact comma count; CSV rows do.
    commas = {line.count(",") for line in core}
    return len(core) >= 3 and len(commas) == 1 and commas != {0}


def _minimize_paragraph(paragraph: list[str], note: Callable[[str], None]) -> list[str]:
    # Keep sentence lines at either edge: with no blank line, the question
    # sits directly above or below what was pasted.
    start, stop = 0, len(paragraph)
    while start < stop and _reads_as_sentence(paragraph[start]):
        start += 1
    while stop > start and _reads_as_sentence(paragraph[stop - 1]):
        stop -= 1
    core = paragraph[start:stop]
    if not core or not _reads_as_data(paragraph, core):
        return paragraph
    note("structured text")
    return [*paragraph[:start], _placeholder("text", "\n".join(core)), *paragraph[stop:]]


def _shape_backstop(lines: list[str], note: Callable[[str], None]) -> list[str]:
    """Remove paragraphs that read as data rather than as sentences.

    Runs last, on what the signatures left. Blank lines and placeholder lines
    both separate paragraphs, so a placeholder is never counted or swallowed.
    """
    output: list[str] = []
    index = 0
    while index < len(lines):
        if not lines[index].strip() or _is_placeholder_line(lines[index]):
            output.append(lines[index])
            index += 1
            continue
        end = _paragraph_end(lines, index)
        output.extend(_minimize_paragraph(lines[index:end], note))
        index = end
    return output


def _structured_document(text: str) -> str | None:
    """A whole message that is a JSON or XML document, rather than a question."""
    stripped = text.strip()
    if stripped.startswith(("<?xml", "<!DOCTYPE", "<html")):
        return "XML/HTML document"
    if stripped[:1] in "{[" and stripped[-1:] in "}]":
        try:
            json.loads(stripped)
        except RecursionError:
            # Too deeply nested to parse is still a document: withhold it rather
            # than let the error escape the minimizer.
            return "JSON document"
        except ValueError:
            return None
        return "JSON document"
    return None


def minimize_for_cloud(text: str) -> tuple[str, list[str]]:
    """Replace pasted machine output with a short description of it.

    Returns the minimized text and the kinds of content that were withheld, in
    first-seen order. Prose is returned unchanged: redaction and truncation
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
        note("code block")
        if "\n" not in match.group(0):
            # A one-line span: the word after the fence is code, not a language,
            # so it must not reach the label.
            return _placeholder("code block", match.group(0).strip("`"))
        # The info string is user text like any other: only a language name
        # from the allowlist reaches the label, so "```hunter2" says nothing.
        language = match.group(1).strip().lower()
        label = f"{language} code block" if language in _FENCE_LANGUAGES else "code block"
        return _placeholder(label, match.group(2).rstrip("\n"))

    def tilde_fence(match: re.Match[str]) -> str:
        note("code block")
        language = match.group(2).strip().lower()
        label = f"{language} code block" if language in _FENCE_LANGUAGES else "code block"
        return _placeholder(label, match.group(3).rstrip("\n"))

    def pem(match: re.Match[str]) -> str:
        note("key material")
        return "[pasted key material omitted]"

    result = _PEM_BLOCK.sub(pem, _TILDE_FENCE.sub(tilde_fence, _FENCE.sub(fence, text)))
    lines = _replace_blocks(result.split("\n"), note)
    return "\n".join(_shape_backstop(lines, note)), kept


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
