"""The deterministic backend: the answer that is always available.

This is the floor of the decision chain. When Jev is unreachable and the local
model is busy or too slow, every request still has to be routed, so this module
is the one thing in the semantic layer that is not allowed to fail, block, or
go anywhere near the network. It reads the request, counts evidence it can name,
and returns in microseconds.

Three commitments shape the code:

* **Inspectable, not clever.** Every point a domain scores comes from a named
  cue: a word list, a phrase list, or a structural signal with a label. The
  signals are exposed (`task_signals`, `step_signals`) so a human, a test, or
  the dashboard can ask *why* a request was called `system_ops` and get an
  answer that is a list of names rather than a weight matrix.

* **Honest numbers.** These are evidence counts normalised to sum to one. They
  are not calibrated probabilities and nothing here pretends otherwise:
  confidence is derived from the margin between the top two scores and is
  capped well below certainty, the probabilistic questions answer in four
  discrete levels rather than inventing two decimal places, and every decision
  carries a note saying it came from rules.

* **No side effects.** Pure functions over the context, no I/O, no state
  between calls, no mutation of the context. `aclose()` exists only to satisfy
  the protocol.

Notes and summaries carry cue *labels* and counts, never fragments of the
request: this backend sees unredacted text and writes nothing durable from it.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from switchboard.config import RulesConfig
from switchboard.decisions.base import (
    Availability,
    BackendFacts,
    StepContext,
    Stopwatch,
    TaskContext,
)
from switchboard.decisions.schema import (
    DIFFICULTY_CRITERIA,
    DOMAINS,
    STEP_OUTCOMES,
    Domain,
    StepDecision,
    StepOutcome,
    TaskDecision,
)

BACKEND_NAME = "rules"

HEURISTIC_NOTE = (
    "rules backend: heuristic cue counts, not calibrated probabilities"
)

_DIFFICULTY_LEVELS = len(DIFFICULTY_CRITERIA)

# --- Lexical cues -----------------------------------------------------------
#
# Every list below is hand-written from the DOMAIN_CRITERIA text in `schema`,
# then extended with the concrete vocabulary that shows up in traffic to this
# machine (a llama.cpp stack driven from PowerShell, with a coding client).
# They are frozensets of *lowercase stems*: `_tokens` strips a trailing plural,
# so "files" finds "file" and "processes" finds "process".
#
# Two rules kept the lists honest while writing them:
#   * a word goes in exactly one domain unless the ambiguity is real;
#   * product names are avoided ("postgres" says nothing about the kind of work
#     being asked for), and generic words that would fire on half of all English
#     ("best", "report", "review") are expressed as phrases instead.

_CODING_WORDS = frozenset(
    {
        "python", "javascript", "typescript", "java", "rust", "golang", "c++", "sql",
        "regex", "function", "class", "method", "variable", "loop", "array", "dict",
        "tuple", "bug", "debug", "debugger", "stacktrace", "traceback", "compile",
        "compiler", "runtime", "syntax", "refactor", "api", "endpoint", "json",
        "yaml", "toml", "repo", "repository", "commit", "merge", "rebase", "branch",
        "pytest", "unittest", "fixture", "mock", "npm", "pip", "uv", "sdk",
        "library", "import", "module", "package", "async", "await", "typing",
        "dataclass", "null", "segfault", "lint", "linter", "mypy", "ruff", "eslint",
        "django", "flask", "fastapi", "react", "numpy", "pandas", "code", "codebase",
        "script", "snippet", "algorithm", "recursion", "pointer", "struct",
        "interface", "generic", "git", "database", "query", "schema", "migration",
        "parser", "serialise", "serialize", "deserialise", "deserialize", "regexp",
        "callback", "closure", "iterator", "decorator", "docstring", "assertion",
        "exception", "stub", "typehint", "coroutine", "threadpool",
    }
)
_CODING_PHRASES = frozenset(
    {
        "unit test", "pull request", "code review", "this code", "the code",
        "stack trace", "type error", "null pointer", "race condition",
        "time complexity", "error handling", "compile error", "merge conflict",
        "write a function", "source file", "test suite", "line of code",
        "does not compile", "throws an", "returns none",
    }
)

_MEDICAL_WORDS = frozenset(
    {
        "symptom", "diagnosis", "diagnose", "patient", "dose", "dosage",
        "prescription", "prescribed", "drug", "medication", "antibiotic",
        "ibuprofen", "paracetamol", "acetaminophen", "metformin", "insulin",
        "statin", "inhibitor", "diabetes", "diabetic", "hypertension",
        "cholesterol", "tumour", "tumor", "nodule", "cancer", "chemotherapy",
        "infection", "inflammation", "fever", "rash", "migraine", "asthma",
        "anaemia", "anemia", "thyroid", "cardiac", "cardiology", "oncology",
        "radiology", "mri", "ecg", "ekg", "biopsy", "haemoglobin", "hemoglobin",
        "platelet", "pathology", "clinical", "contraindication", "surgery",
        "surgical", "vaccine", "vaccination", "pneumonia", "sepsis", "stroke",
        "fracture", "arthritis", "allergy", "allergic", "pregnancy", "pregnant",
        "paediatric", "pediatric", "dermatitis", "triage", "doctor", "physician",
        "nurse", "clinic", "hospital", "prognosis", "malignant", "benign",
    }
)
_MEDICAL_PHRASES = frozenset(
    {
        "blood pressure", "side effect", "side effects", "lab result",
        "lab results", "blood test", "heart rate", "medical history", "ct scan",
        "white blood cell", "drug interaction", "chest pain",
        "shortness of breath", "clinical trial", "differential diagnosis",
        "see a doctor", "my doctor", "is it safe to take", "mg of",
    }
)

_WRITING_WORDS = frozenset(
    {
        "essay", "email", "letter", "draft", "rewrite", "reword", "rephrase",
        "phrasing", "proofread", "paragraph", "sentence", "tone", "wording",
        "prose", "story", "poem", "blog", "article", "newsletter", "headline",
        "caption", "grammar", "punctuation", "typo", "narrative", "dialogue",
        "manuscript", "chapter", "formal", "informal", "polite", "apology",
        "invitation", "announcement", "eulogy", "toast", "byline", "copywriting",
    }
)
_WRITING_PHRASES = frozenset(
    {
        "write an email", "draft an email", "make it sound", "more concise",
        "cover letter", "thank you note", "subject line", "friendly tone",
        "short story", "blog post", "press release", "the wording",
        "sounds less", "sounds more", "rewrite this", "in my own words",
    }
)

_RESEARCH_WORDS = frozenset(
    {
        "compare", "comparison", "versus", "alternative", "option", "pros",
        "cons", "source", "citation", "cite", "literature", "study", "studies",
        "evidence", "survey", "whitepaper", "arxiv", "pubmed", "doi", "finding",
        "consensus", "landscape", "vendor", "competitor", "benchmarking",
        "methodology", "reference", "bibliography",
    }
)
_RESEARCH_PHRASES = frozenset(
    {
        "pros and cons", "which is better", "peer reviewed", "cite sources",
        "with sources", "latest research", "state of the art",
        "literature review", "trade offs", "look up", "research on",
        "the studies say", "according to the", "market share",
    }
)

_SYSTEM_OPS_WORDS = frozenset(
    {
        "powershell", "cmd", "terminal", "console", "shell", "bash", "explorer",
        "registry", "regedit", "service", "process", "pid", "taskkill",
        "tasklist", "netstat", "ipconfig", "winget", "choco", "chocolatey",
        "msi", "installer", "install", "uninstall", "reboot", "shutdown",
        "startup", "boot", "scheduler", "schtask", "cron", "systemctl",
        "daemon", "driver", "firewall", "defender", "disk", "drive",
        "partition", "folder", "directory", "permission", "admin",
        "administrator", "sudo", "chmod", "symlink", "gpu", "cpu", "ram",
        "vram", "nvidia", "wsl", "port", "localhost", "ssh", "robocopy",
        "xcopy", "unzip", "backup", "log", "logfile", "windows", "filesystem",
        "file", "clipboard", "desktop", "reinstall", "uptime", "throttle",
    }
)
_SYSTEM_OPS_PHRASES = frozenset(
    {
        "task manager", "device manager", "event viewer", "command prompt",
        "environment variable", "run as administrator", "free up space",
        "kill the process", "open the folder", "file explorer", "control panel",
        "start menu", "scheduled task", "disk space", "windows update",
        "group policy", "in the terminal", "on my machine", "on this computer",
        "path variable", "hard drive",
    }
)

# `general` is the catch-all, so it gets a standing prior instead of a big list.
# These are only the cues that actively argue for "plain conversation", and they
# are weighted below a domain cue on purpose.
_GENERAL_WORDS = frozenset(
    {
        "explain", "define", "definition", "meaning", "translate", "joke",
        "recipe", "weather", "trivia", "opinion", "analogy", "beginner",
        "layman", "simply",
    }
)
_GENERAL_PHRASES = frozenset(
    {
        "in simple terms", "like i am five", "what is the difference",
        "tell me about", "how does it work",
    }
)

_LEXICON: dict[Domain, tuple[frozenset[str], frozenset[str]]] = {
    "general": (_GENERAL_WORDS, _GENERAL_PHRASES),
    "coding": (_CODING_WORDS, _CODING_PHRASES),
    "medical": (_MEDICAL_WORDS, _MEDICAL_PHRASES),
    "writing": (_WRITING_WORDS, _WRITING_PHRASES),
    "research": (_RESEARCH_WORDS, _RESEARCH_PHRASES),
    "system_ops": (_SYSTEM_OPS_WORDS, _SYSTEM_OPS_PHRASES),
}

# A general cue is weak evidence (the words also appear in every other domain),
# so it scores below a domain word; a phrase is worth more than a word because
# it cannot fire by accident.
_WORD_WEIGHT = 1.0
_GENERAL_WORD_WEIGHT = 0.7
_PHRASE_WEIGHT = 1.6
_GENERAL_PRIOR = 0.9
# Smoothing floor: without it a domain that scores nothing would take 0% of the
# normalised vector, which would read as certainty we do not have.
_SCORE_FLOOR = 0.15

# Commands that only make sense as "operate this computer". Language tooling
# (git, python, pip, npm) deliberately lives in the coding list instead.
_SHELL_COMMANDS = frozenset(
    {
        "ls", "cd", "dir", "cat", "grep", "rm", "mv", "cp", "chmod", "chown",
        "sudo", "apt", "brew", "curl", "wget", "ssh", "scp", "tar", "ps",
        "kill", "df", "du", "mkdir", "touch", "echo", "taskkill", "tasklist",
        "netstat", "ipconfig", "sfc", "chkdsk", "diskpart", "schtasks",
        "regedit", "powershell", "pwsh", "cmd", "wsl", "systemctl",
        "journalctl", "winget", "choco", "robocopy", "xcopy", "docker",
        "nvidia-smi",
    }
)

# Verbs that reach outside the conversation. "write" and "read" are absent on
# purpose: "write a poem" is not a tool call.
_ACTION_VERBS = frozenset(
    {
        "run", "open", "launch", "install", "uninstall", "reinstall", "delete",
        "remove", "move", "copy", "rename", "kill", "start", "stop", "restart",
        "download", "upload", "fetch", "list", "find", "check", "backup",
        "unzip", "mount", "clear", "clean", "deploy", "execute", "search",
    }
)

_IMPERATIVE_OPENERS = frozenset(
    _ACTION_VERBS
    | {
        "write", "fix", "create", "make", "build", "add", "update", "draft",
        "summarise", "summarize", "compare", "explain", "show", "review",
        "refactor", "debug", "translate", "configure", "generate", "implement",
        "design", "analyse", "analyze", "convert", "help", "rewrite", "tell",
        "give", "please",
    }
)

_GREETINGS = frozenset(
    {
        "hi", "hello", "hey", "yo", "howdy", "morning", "afternoon", "evening",
        "gm", "sup", "thanks", "thank", "thx", "ty", "ok", "okay", "cool",
        "nice", "great", "yes", "no", "yep", "nope", "bye", "goodbye", "cheers",
        "there", "you", "much", "again", "good", "that", "it", "welcome",
        "sorry", "please", "claude", "switchboard", "u",
    }
)

# Replies that close a turn rather than ask for work. Matched as phrases so
# "thanks" counts but "thanks for the detailed spec, now ..." does not.
_ACK_PHRASES = frozenset(
    {"thanks", "thank you", "that worked", "exactly what i needed", "got it",
     "never mind", "no worries", "perfect", "makes sense", "all good"}
)

_VAGUE_WORDS = frozenset(
    {"it", "this", "that", "them", "thing", "stuff", "something", "somehow",
     "whatever", "better", "broken", "wrong", "same", "one"}
)

_HARD_WORDS = frozenset(
    {
        "prove", "derive", "design", "architecture", "optimise", "optimize",
        "concurrency", "deadlock", "scalability", "complexity", "cryptography",
        "tradeoff", "tradeoffs", "refactor", "throughput", "latency",
        "consistency", "idempotent", "invariant", "rationale", "justify",
    }
)
_HARD_PHRASES = frozenset(
    {
        "why does", "why is", "trade off", "trade offs", "root cause",
        "race condition", "thread safe", "from scratch", "end to end",
        "pros and cons", "best way to", "how would you", "what are the risks",
        "explain why", "compare the",
    }
)

_MULTIPART_PHRASES = frozenset(
    {"and then", "after that", "as well as", "once that", "then write",
     "then run", "and also", "followed by"}
)

# Question frames that ask for knowledge *about* an action rather than the
# action itself. They cap tool_need rather than adding to it.
_KNOWLEDGE_FRAMES = frozenset(
    {"how do i", "how can i", "how would i", "how do you", "what is", "what s",
     "whats", "what does", "what are", "why is", "why does", "is it possible",
     "should i", "can you explain", "explain how", "tell me how",
     "what happens when", "difference between"}
)

_LOOKUP_CUES = frozenset(
    {"look up", "search for", "latest", "current", "today", "right now",
     "cite sources", "with sources", "up to date", "in the news", "this week"}
)

# Tool names are evidence about what the *client* can do, not about what the
# user asked for, so their total contribution is capped below a single phrase
# cue: they nudge a tie, they never decide a domain.
_TOOL_HINTS: dict[str, Domain] = {
    "file": "system_ops", "files": "system_ops", "filesystem": "system_ops",
    "dir": "system_ops", "folder": "system_ops", "path": "system_ops",
    "shell": "system_ops", "bash": "system_ops", "powershell": "system_ops",
    "terminal": "system_ops", "command": "system_ops", "process": "system_ops",
    "registry": "system_ops", "screenshot": "system_ops", "click": "system_ops",
    "keyboard": "system_ops", "window": "system_ops", "service": "system_ops",
    "app": "system_ops", "clipboard": "system_ops",
    "search": "research", "web": "research", "browse": "research",
    "fetch": "research", "url": "research", "http": "research",
    "wiki": "research", "scholar": "research", "arxiv": "research",
    "pubmed": "research",
    "code": "coding", "repo": "coding", "git": "coding", "lint": "coding",
    "test": "coding", "pytest": "coding", "compile": "coding",
}
_TOOL_HINT_WEIGHT = 0.6
_TOOL_HINT_CAP = 1.8

# --- Structural patterns ----------------------------------------------------

_TOKEN = re.compile(r"[a-z][a-z0-9+#]*|[0-9]+")
_NON_WORD = re.compile(r"[^a-z0-9]+")

_CODE_FENCE = re.compile(r"```|~~~")
_INLINE_CODE = re.compile(r"`[^`\n]{2,}`")
_TRACEBACK = re.compile(r"(?i)traceback \(most recent call last\)")
_STACK_FRAME = re.compile(r'^\s*(?:File "[^"]+", line \d+|at [\w$.<>]+\()', re.MULTILINE)
# Case-sensitive on purpose: "ValueError" is a class name, "an error" is prose.
_EXCEPTION_NAME = re.compile(r"\b[A-Z][A-Za-z]{2,}(?:Error|Exception)\b")
_DIFF_MARKER = re.compile(r"^(?:\+\+\+ |--- |@@ |diff --git )", re.MULTILINE)
_WINDOWS_PATH = re.compile(r"(?i)\b[a-z]:[\\/][\w .$-]+[\\/]?|\\\\[\w.-]+\\[\w.-]+")
_POSIX_PATH = re.compile(r"""(?:^|[\s"'(])(?:~|\.{1,2})?/[\w.-]+(?:/[\w.-]+)+""")
_URL = re.compile(r"(?i)\bhttps?://\S+|\bwww\.[\w.-]{4,}")
_CMDLET = re.compile(
    r"(?i)\b(?:get|set|new|remove|start|stop|restart|test|invoke|select|where)-[a-z]{3,}\b"
)
_LIST_ITEM = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s+\S", re.MULTILINE)
_FILE_EXT = re.compile(r"(?i)\b[\w.-]+\.([a-z0-9]{1,5})\b")

_CODE_EXTENSIONS = frozenset(
    {"py", "pyi", "js", "ts", "tsx", "jsx", "rs", "go", "java", "rb", "php",
     "c", "cpp", "h", "hpp", "cs", "sql", "html", "css", "json", "toml",
     "yaml", "yml", "md", "ipynb", "tf", "proto"}
)
_OPS_EXTENSIONS = frozenset(
    {"ps1", "bat", "cmd", "sh", "exe", "dll", "msi", "zip", "log", "ini",
     "cfg", "reg", "sys", "gguf", "iso", "lnk"}
)

# Structural signal -> how much it argues for each domain. Written out rather
# than computed so the table can be read and argued with.
_STRUCTURAL_WEIGHTS: dict[str, dict[Domain, float]] = {
    "code-fence": {"coding": 3.0},
    "inline-code": {"coding": 1.2, "system_ops": 0.4},
    "traceback": {"coding": 3.0},
    "stack-frame": {"coding": 2.0},
    "exception-name": {"coding": 1.5},
    "diff": {"coding": 2.5},
    "windows-path": {"system_ops": 1.6, "coding": 0.6},
    "posix-path": {"system_ops": 0.8, "coding": 1.0},
    "code-file": {"coding": 1.5},
    "ops-file": {"system_ops": 1.2},
    "shell-command": {"system_ops": 1.4},
    "powershell-cmdlet": {"system_ops": 2.0},
    "url": {"research": 1.2},
}


# --- Task analysis ----------------------------------------------------------


@dataclass(slots=True)
class TaskSignals:
    """Everything the rules saw, by name. Safe to log: labels and counts only."""

    scores: dict[str, float] = field(default_factory=dict)
    lexical_hits: dict[str, int] = field(default_factory=dict)
    structural: list[str] = field(default_factory=list)
    tool_hits: dict[str, int] = field(default_factory=dict)
    word_count: int = 0
    has_code: bool = False
    has_traceback: bool = False
    has_path: bool = False
    has_shell: bool = False
    has_url: bool = False
    has_lists: bool = False
    action_command: bool = False
    knowledge_frame: bool = False
    lookup_cue: bool = False
    # Greetings, acknowledgements and one-word replies: a turn, but not a task.
    small_talk: bool = False
    question: bool = False
    vague_only: bool = False

    def top_domains(self, count: int = 2) -> list[tuple[str, float]]:
        return sorted(self.scores.items(), key=lambda item: (-item[1], item[0]))[:count]


def _tokens(lowered: str) -> frozenset[str]:
    """Word tokens plus a deliberately crude plural stem.

    A real stemmer would be a dependency; the cue lists are hand-written, so
    stripping a trailing "s"/"es" buys most of the coverage for none of the risk.
    """
    found = set(_TOKEN.findall(lowered))
    for token in list(found):
        if len(token) > 5 and token.endswith("es"):
            found.add(token[:-2])
        if len(token) > 4 and token.endswith("s"):
            found.add(token[:-1])
    return frozenset(found)


def _phrase_text(lowered: str) -> str:
    """Punctuation-free, space-padded text so phrases match on word boundaries."""
    return f" {_NON_WORD.sub(' ', lowered).strip()} "


def _has_phrase(phrase_text: str, phrases: frozenset[str]) -> list[str]:
    return sorted(phrase for phrase in phrases if f" {phrase} " in phrase_text)


def _action_in_command_position(lowered: str) -> bool:
    """An action verb used as a command, not as a noun.

    "what's the difference between a list and a tuple" contains "list"; "list the
    running processes" starts with it. Only the second is a request to act, so
    the verb has to sit at the start of the message or of a clause.
    """
    for verb in _ACTION_VERBS:
        pattern = re.compile(
            r"(?:^|[.!?\n]\s*|,\s*|\band then\s+|\bthen\s+|\band\s+|\bplease\s+"
            r"|\bcan you\s+|\bcould you\s+|\bgo ahead and\s+|^\s*[-*\d.)]+\s*)"
            + re.escape(verb)
            + r"\b",
            re.IGNORECASE,
        )
        if pattern.search(lowered):
            return True
    return False


def task_signals(context: TaskContext) -> TaskSignals:
    """Score the six domains from named cues. Pure; the context is not touched."""
    message = context.message or ""
    lowered = message.lower()
    phrase_text = _phrase_text(lowered)
    tokens = _tokens(lowered)
    words = lowered.split()

    scores: dict[str, float] = {domain: 0.0 for domain in DOMAINS}
    scores["general"] = _GENERAL_PRIOR
    lexical_hits: dict[str, int] = {}

    for domain, (word_cues, phrase_cues) in _LEXICON.items():
        matched_words = tokens & word_cues
        matched_phrases = _has_phrase(phrase_text, phrase_cues)
        weight = _GENERAL_WORD_WEIGHT if domain == "general" else _WORD_WEIGHT
        scores[domain] += len(matched_words) * weight
        scores[domain] += len(matched_phrases) * _PHRASE_WEIGHT
        hits = len(matched_words) + len(matched_phrases)
        if hits:
            lexical_hits[domain] = hits

    structural: list[str] = []

    def fire(name: str) -> None:
        structural.append(name)
        for domain, weight in _STRUCTURAL_WEIGHTS[name].items():
            scores[domain] += weight

    has_fence = bool(_CODE_FENCE.search(message))
    if has_fence:
        fire("code-fence")
    if _INLINE_CODE.search(message):
        fire("inline-code")
    has_traceback = bool(_TRACEBACK.search(message))
    if has_traceback:
        fire("traceback")
    if _STACK_FRAME.search(message):
        fire("stack-frame")
        has_traceback = True
    if _EXCEPTION_NAME.search(message):
        fire("exception-name")
    if _DIFF_MARKER.search(message):
        fire("diff")

    has_windows_path = bool(_WINDOWS_PATH.search(message))
    if has_windows_path:
        fire("windows-path")
    has_posix_path = bool(_POSIX_PATH.search(message))
    if has_posix_path:
        fire("posix-path")

    extensions = {match.lower() for match in _FILE_EXT.findall(message)}
    if extensions & _CODE_EXTENSIONS:
        fire("code-file")
    has_ops_file = bool(extensions & _OPS_EXTENSIONS)
    if has_ops_file:
        fire("ops-file")

    has_shell = bool(tokens & _SHELL_COMMANDS)
    if has_shell:
        fire("shell-command")
    has_cmdlet = bool(_CMDLET.search(message))
    if has_cmdlet:
        fire("powershell-cmdlet")
        has_shell = True

    has_url = bool(_URL.search(message))
    if has_url:
        fire("url")

    tool_hits: dict[str, int] = {}
    tool_budget = _TOOL_HINT_CAP
    # Attached tools say what the client *could* do, not what this message asks
    # for. So they may reinforce a domain the words already support, and may
    # decide only when the words support nothing at all. Without that rule a
    # client which always attaches a filesystem tool would turn every question,
    # including "why is the sky blue?", into system operations.
    # `general` carries a standing prior, so it does not count as evidence here;
    # otherwise every message would look "supported" and tools could never break
    # a genuine tie. A knowledge frame ("why is ...", "what does ...") does count:
    # a well-formed question is evidence that no action was requested, which is
    # what keeps "why is the sky blue?" out of system_ops when a shell tool is
    # attached.
    knowledge_frame = any(f" {frame} " in phrase_text for frame in _KNOWLEDGE_FRAMES)
    lexically_supported = {
        domain
        for domain, score in scores.items()
        if score > (_GENERAL_PRIOR if domain == "general" else 0.0)
    }
    if knowledge_frame:
        lexically_supported.add("general")
    for name in context.tool_names:
        parts = {part for part in _NON_WORD.sub(" ", name.lower()).split() if part}
        for part in sorted(parts):
            domain = _TOOL_HINTS.get(part)
            if domain is None or tool_budget <= 0:
                continue
            if lexically_supported and domain not in lexically_supported:
                continue
            awarded = min(_TOOL_HINT_WEIGHT, tool_budget)
            scores[domain] += awarded
            tool_budget -= awarded
            tool_hits[domain] = tool_hits.get(domain, 0) + 1

    content_tokens = {token for token in tokens if not token.isdigit()}
    greeting_only = bool(content_tokens) and content_tokens <= _GREETINGS and len(words) <= 6
    acknowledgement = (
        len(words) <= 10
        and not lexical_hits
        and not structural
        and bool(_has_phrase(phrase_text, _ACK_PHRASES))
    )

    return TaskSignals(
        scores=scores,
        lexical_hits=lexical_hits,
        structural=structural,
        tool_hits=tool_hits,
        word_count=len(words),
        has_code=has_fence or has_traceback or bool(_INLINE_CODE.search(message)),
        has_traceback=has_traceback,
        has_path=has_windows_path or has_posix_path,
        has_shell=has_shell,
        has_url=has_url,
        has_lists=len(_LIST_ITEM.findall(message)) >= 2,
        action_command=_action_in_command_position(lowered),
        knowledge_frame=knowledge_frame,
        lookup_cue=any(f" {cue} " in phrase_text for cue in _LOOKUP_CUES),
        small_talk=greeting_only or acknowledgement,
        question="?" in message,
        vague_only=bool(content_tokens) and content_tokens <= (_VAGUE_WORDS | _IMPERATIVE_OPENERS),
    )


def _normalise(scores: dict[str, float]) -> dict[str, float]:
    """Floor, then scale to sum to one."""
    floored = {key: max(value, _SCORE_FLOOR) for key, value in scores.items()}
    total = sum(floored.values()) or 1.0
    return {key: value / total for key, value in floored.items()}


def _margin_confidence(probabilities: dict[str, float]) -> float:
    """Confidence from the gap between the best and second-best option.

        confidence = 0.20 + 0.55 * (p_top - p_second) / p_top

    Zero margin (a tie) gives 0.20; a runner-up at the smoothing floor
    approaches, but never reaches, 0.75. The ceiling is the point: a keyword
    count is never entitled to claim certainty, so the chain can always tell a
    rules answer from a model's.
    """
    ranked = sorted(probabilities.values(), reverse=True)
    if not ranked:
        return 0.2
    top = ranked[0]
    second = ranked[1] if len(ranked) > 1 else 0.0
    if top <= 0:
        return 0.2
    return round(0.20 + 0.55 * ((top - second) / top), 4)


def _difficulty(signals: TaskSignals, domain: Domain, lowered: str, phrase_text: str) -> tuple[float, int]:
    """Difficulty on the 0-3 scale, plus how many cues moved it.

    Starts at ROUTINE and is argued up or down; the cue count feeds confidence,
    because a score built from no evidence deserves less of it.
    """
    if signals.small_talk:
        return 0.15, 1

    score = 1.0
    used = 0

    words = signals.word_count
    if words <= 5:
        score -= 0.45
        used += 1
    elif words <= 12:
        score -= 0.15
        used += 1
    elif words >= 100:
        score += 0.6
        used += 1
    elif words >= 40:
        score += 0.35
        used += 1

    clauses = lowered.count(",") + lowered.count(";") + phrase_text.count(" but ")
    if clauses:
        score += min(0.45, 0.15 * clauses)
        used += 1

    multipart = signals.has_lists or any(
        f" {phrase} " in phrase_text for phrase in _MULTIPART_PHRASES
    )
    if multipart:
        score += 0.5
        used += 1

    if signals.has_code:
        score += 0.5
        used += 1
    if signals.has_traceback:
        # A real error in hand means debugging, which the criteria call moderate.
        score += 0.35
        used += 1

    tokens = _tokens(lowered)
    hard = bool(tokens & _HARD_WORDS) or any(
        f" {phrase} " in phrase_text for phrase in _HARD_PHRASES
    )
    if hard:
        score += 0.8
        used += 1

    if domain == "medical":
        # Clinical content needs domain knowledge even when the question is short.
        score += 0.3
        used += 1

    if signals.question and words <= 8 and not hard:
        score -= 0.2
        used += 1

    return max(0.0, min(3.0, score)), used


def _difficulty_probabilities(value: float) -> dict[str, float]:
    """Split the mass between the two levels the float sits between.

    Chosen so `score_from_probabilities` returns the float back unchanged: the
    distribution is a presentation of the score, not a second opinion.
    """
    low = max(0, min(_DIFFICULTY_LEVELS - 1, int(math.floor(value))))
    high = min(low + 1, _DIFFICULTY_LEVELS - 1)
    probabilities = {str(level): 0.0 for level in range(_DIFFICULTY_LEVELS)}
    if high == low:
        probabilities[str(low)] = 1.0
        return probabilities
    upper = value - low
    probabilities[str(low)] = 1.0 - upper
    probabilities[str(high)] = upper
    return probabilities


# Four levels, reached by explicit rules. Anything finer would be invented.
_LEVELS = (0.1, 0.35, 0.6, 0.85)


def _tool_need(signals: TaskSignals, domain: Domain, tools_attached: bool) -> float:
    """How likely the request needs an action outside the conversation.

    Points for evidence, then a ladder onto four levels. A knowledge frame
    ("how do I ...") caps the result: asking *about* a command is not running it.
    """
    points = 0
    if signals.has_path:
        points += 2
    if signals.has_shell:
        points += 2
    if signals.action_command:
        points += 2
    if domain == "system_ops":
        points += 1
    if tools_attached and signals.action_command:
        points += 1
    if signals.has_url or signals.lookup_cue:
        points += 1

    if points >= 4:
        level = 0.85
    elif points >= 2:
        level = 0.6
    elif points >= 1:
        level = 0.35
    else:
        level = 0.1

    if signals.knowledge_frame:
        level = min(level, 0.35)
    return level


def _ambiguity(signals: TaskSignals, context: TaskContext) -> float:
    """How likely a competent assistant would have to ask before starting."""
    if signals.small_talk:
        # A greeting is not ambiguous; it just is not a task.
        return 0.1

    concrete = signals.has_path or signals.has_code or signals.has_url or signals.lexical_hits
    if signals.word_count <= 3 and not concrete:
        level = 0.85
    elif signals.vague_only and signals.word_count <= 8:
        level = 0.85
    elif signals.question and signals.word_count >= 4:
        # A question mark plus a subject: short is not the same as unclear.
        level = 0.1
    elif signals.word_count <= 6 and not concrete and not signals.structural:
        level = 0.6
    elif signals.word_count <= 15 and not concrete and not signals.action_command:
        level = 0.35
    else:
        level = 0.1

    if context.prior_message and level > 0.1:
        # The previous turn usually supplies the referent a bare "it" is missing.
        level = _LEVELS[max(0, _LEVELS.index(level) - 1)]
    return level


# --- Step analysis ----------------------------------------------------------

# Counts that say nothing went wrong must not trip the error patterns; pytest's
# "4 passed, 0 failed" is the common case on this machine.
_BENIGN_COUNTS = re.compile(
    r"(?i)\b(?:0|no|zero)\s+(?:errors?|failures?|failed|warnings?)\b"
    r"|\b(?:errors?|failures?|failed)\s*[:=]\s*0\b"
)

_ERROR_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("traceback", re.compile(r"(?i)traceback \(most recent call last\)")),
    ("stack-frame", re.compile(r'^\s*(?:File "[^"]+", line \d+|at [\w$.<>]+\()', re.MULTILINE)),
    ("exception-name", re.compile(r"\b[A-Z][A-Za-z]{2,}(?:Error|Exception)\b")),
    ("exception", re.compile(r"(?i)\bexception\b")),
    ("error-label", re.compile(r"^\s*error\b|\berrors?\s*[:!]", re.IGNORECASE | re.MULTILINE)),
    (
        "not-found",
        re.compile(
            r"(?i)\bnot found\b|\bno such file\b|\bis not recognized\b"
            r"|\bcannot find\b|\bcommand not found\b|\bmissing\b"
        ),
    ),
    (
        "permission",
        re.compile(r"(?i)\bpermission denied\b|\baccess is denied\b|\bunauthori[sz]ed\b|\bforbidden\b"),
    ),
    ("failed", re.compile(r"(?i)\bfail(?:ed|ure|ures|s)?\b|\bfatal\b|\baborted\b")),
    ("nonzero-exit", re.compile(r"(?i)\bexit (?:code|status)\s*[1-9]\d*\b")),
    (
        "timeout",
        re.compile(r"(?i)\btimed out\b|\btimeout\b|\bconnection reset\b|\btemporarily unavailable\b"),
    ),
)

_SUCCESS_MARKERS = re.compile(
    r"(?i)\b(?:success|successful|successfully|succeeded|completed|complete|done"
    r"|created|installed|updated|saved|wrote|written|passed|finished|added"
    r"|removed|ok)\b"
)
_PARTIAL_MARKERS = re.compile(r"(?i)\bpartial(?:ly)?\b|\b\d+ passed,\s*[1-9]\d* failed\b")
_COMPLETION_MARKERS = re.compile(
    r"(?i)\ball (?:tests|checks) passed\b|\b(?:task|work|job) (?:is )?completed?\b"
    r"|\bnothing (?:left|else) to do\b|\bfinished successfully\b"
)

# Failures the same command cannot recover from vs. failures worth one retry.
_INHERENT_ERRORS = frozenset({"not-found", "permission"})
_TRANSIENT_ERRORS = frozenset({"timeout"})

_STOPWORDS = frozenset(
    {"the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
     "is", "are", "was", "were", "be", "it", "this", "that", "then", "so",
     "at", "by", "from", "as", "into", "if", "not", "no", "all", "any"}
)


@dataclass(slots=True)
class StepSignals:
    """Named evidence from one finished step."""

    errors: list[str] = field(default_factory=list)
    successes: bool = False
    partial: bool = False
    completion_marker: bool = False
    empty_result: bool = False
    exit_code: int | None = None
    repeated: bool = False
    goal_coverage: float = 0.0


def _content_tokens(text: str) -> frozenset[str]:
    return frozenset(token for token in _tokens(text.lower()) if token not in _STOPWORDS)


def step_signals(context: StepContext) -> StepSignals:
    """Read the hard facts and the named error/success patterns. Pure."""
    result = context.result or ""
    scannable = _BENIGN_COUNTS.sub(" ok ", result)

    errors = [name for name, pattern in _ERROR_PATTERNS if pattern.search(scannable)]

    action = " ".join((context.action or "").split()).casefold()
    repeated = bool(action) and any(
        action == " ".join(previous.split()).casefold()
        for previous in context.previous_attempts
    )

    goal_tokens = _content_tokens(context.goal or "")
    result_tokens = _content_tokens(result)
    coverage = len(goal_tokens & result_tokens) / len(goal_tokens) if goal_tokens else 0.0

    return StepSignals(
        errors=errors,
        successes=bool(_SUCCESS_MARKERS.search(scannable)),
        partial=bool(_PARTIAL_MARKERS.search(result)),
        completion_marker=bool(_COMPLETION_MARKERS.search(result)),
        empty_result=not result.strip(),
        exit_code=context.exit_code,
        repeated=repeated,
        goal_coverage=coverage,
    )


def _outcome_probabilities(outcome: StepOutcome, strength: float) -> dict[str, float]:
    """Give the chosen outcome `strength`, and hand most of the doubt to 'unclear'.

    When these rules are wrong it is almost always because the result text
    carried evidence they do not recognise, which is exactly what 'unclear'
    means - so the residual is not spread evenly.
    """
    strength = max(0.3, min(0.9, strength))
    probabilities = {name: 0.0 for name in STEP_OUTCOMES}
    probabilities[outcome] = strength
    rest = 1.0 - strength
    others = [name for name in STEP_OUTCOMES if name != outcome]
    if outcome != "unclear":
        probabilities["unclear"] = rest * 0.5
        remaining = [name for name in others if name != "unclear"]
    else:
        remaining = others
    share = (rest - probabilities["unclear"] if outcome != "unclear" else rest) / len(remaining)
    for name in remaining:
        probabilities[name] = share
    return probabilities


# --- The backend ------------------------------------------------------------


class RulesBackend:
    """Deterministic last resort. Always available, never raises, never waits.

    `RulesConfig.enabled` is the chain's switch for whether this backend is
    constructed at all; once it exists it reports `ready` unconditionally,
    because a fallback that can decline is not a fallback.
    """

    name = BACKEND_NAME

    def __init__(self, config: RulesConfig | None = None) -> None:
        self.config = config or RulesConfig()

    async def availability(self, facts: BackendFacts) -> Availability:
        # `facts` is ignored on purpose: no server, no model, no network.
        return Availability.ok()

    async def decide_task(self, context: TaskContext, facts: BackendFacts) -> TaskDecision:
        timer = Stopwatch()
        try:
            return self._task(context, timer)
        except Exception as error:  # pragma: no cover - the last resort cannot raise
            return TaskDecision(
                backend=BACKEND_NAME,
                latency_ms=timer.ms,
                degraded=True,
                notes=[HEURISTIC_NOTE, f"rules failed internally ({type(error).__name__}); defaults used"],
            )

    def _task(self, context: TaskContext, timer: Stopwatch) -> TaskDecision:
        signals = task_signals(context)
        probabilities = _normalise(signals.scores)
        domain = max(probabilities, key=lambda key: (probabilities[key], key == "general"))
        confidence = _margin_confidence(probabilities)

        lowered = (context.message or "").lower()
        difficulty, cues_used = _difficulty(signals, domain, lowered, _phrase_text(lowered))
        # The rules see few difficulty cues, so confidence stays in a low band:
        # 0.25 with nothing to go on, 0.60 with four or more cues in agreement.
        difficulty_confidence = round(0.25 + 0.35 * min(1.0, cues_used / 4), 4)

        top = signals.top_domains(2)
        notes = [
            HEURISTIC_NOTE,
            "domain cues: " + ", ".join(f"{name}={score:.1f}" for name, score in top),
        ]
        if signals.structural:
            notes.append("structural: " + ", ".join(dict.fromkeys(signals.structural)))
        if signals.tool_hits:
            notes.append(
                "tool cues: " + ", ".join(f"{name}x{count}" for name, count in sorted(signals.tool_hits.items()))
            )

        return TaskDecision(
            domain=domain,  # type: ignore[arg-type]
            domain_probabilities=probabilities,
            domain_confidence=confidence,
            difficulty=difficulty,
            difficulty_probabilities=_difficulty_probabilities(difficulty),
            difficulty_confidence=difficulty_confidence,
            tool_need=_tool_need(signals, domain, bool(context.tool_names)),  # type: ignore[arg-type]
            ambiguity=_ambiguity(signals, context),
            backend=BACKEND_NAME,
            latency_ms=timer.ms,
            notes=notes,
        )

    async def decide_step(self, context: StepContext, facts: BackendFacts) -> StepDecision:
        timer = Stopwatch()
        try:
            return self._step(context, timer)
        except Exception as error:  # pragma: no cover - the last resort cannot raise
            return StepDecision(
                backend=BACKEND_NAME,
                latency_ms=timer.ms,
                degraded=True,
                notes=[HEURISTIC_NOTE, f"rules failed internally ({type(error).__name__}); defaults used"],
            )

    def _step(self, context: StepContext, timer: Stopwatch) -> StepDecision:
        signals = step_signals(context)
        notes = [HEURISTIC_NOTE]

        outcome, strength, progress = _classify_step(signals, notes)

        needs_new_approach = _new_approach(signals, context, outcome, notes)
        if signals.repeated:
            progress = min(progress, 0.1)

        complete = 0.05
        if outcome == "success":
            # A rule engine sees one step, never the whole task, so it does not
            # get to say the goal is met: 0.4 is the most it may claim, and only
            # when the result both says so and mentions what the goal asked for.
            complete = 0.2
            if signals.completion_marker and signals.goal_coverage >= 0.5:
                complete = 0.4
        elif outcome == "partial":
            complete = 0.1

        probabilities = _outcome_probabilities(outcome, strength)
        return StepDecision(
            outcome=outcome,
            outcome_probabilities=probabilities,
            outcome_confidence=_margin_confidence(probabilities),
            progress=progress,
            complete=complete,
            needs_new_approach=needs_new_approach,
            backend=BACKEND_NAME,
            latency_ms=timer.ms,
            notes=notes,
        )

    async def aclose(self) -> None:
        """Nothing to close: no client, no socket, no file."""
        return None


def _classify_step(signals: StepSignals, notes: list[str]) -> tuple[StepOutcome, float, float]:
    """Hard facts first, then named text patterns."""
    if signals.exit_code is not None and signals.exit_code != 0:
        notes.append(f"exit code {signals.exit_code}")
        return "failure", 0.9, 0.15

    if signals.empty_result:
        # A silent success and a silent no-op look identical from here.
        notes.append("empty result")
        return "unclear", 0.5, 0.5

    if signals.errors:
        notes.append("error cues: " + ", ".join(signals.errors))
        if signals.partial or signals.successes:
            return "partial", 0.55, 0.55
        return "failure", 0.85 if len(signals.errors) > 1 else 0.75, 0.15

    if signals.partial:
        notes.append("partial marker")
        return "partial", 0.6, 0.55

    if signals.successes:
        notes.append("success markers")
        if signals.exit_code == 0:
            return "success", 0.85, 0.85
        return "success", 0.6, 0.75

    if signals.exit_code == 0:
        notes.append("exit code 0 without explicit markers")
        return "success", 0.55, 0.7

    notes.append("no recognised cues in the result")
    return "unclear", 0.5, 0.5


def _new_approach(
    signals: StepSignals, context: StepContext, outcome: StepOutcome, notes: list[str]
) -> float:
    """Whether repeating the same action unchanged would fail again."""
    if signals.repeated:
        notes.append("identical attempt already tried")
        return 0.9

    if outcome == "success":
        return 0.05
    if outcome == "partial":
        return 0.4
    if outcome == "unclear":
        return 0.5

    kinds = set(signals.errors)
    if kinds & _INHERENT_ERRORS:
        notes.append("obstacle looks inherent to this approach")
        return 0.8
    if kinds & _TRANSIENT_ERRORS:
        notes.append("obstacle looks transient")
        value = 0.25
    else:
        value = 0.55

    if context.attempt >= 3:
        # Three goes at the same step: whatever the error text claims, "it was
        # bad luck" has stopped being the better explanation.
        notes.append(f"attempt {context.attempt} of the same step")
        value = max(value, 0.6)
    return value
