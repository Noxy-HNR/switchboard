"""What leaves the machine: minimization, redaction and truncation.

`minimize_for_cloud` decides what a cloud provider may see at all, so both
directions matter. Machine output that slips through is a leak; prose that is
wrongly removed silently degrades every decision Jev makes. Each case below is
one of those two failures.
"""

from __future__ import annotations

import pytest

from switchboard.privacy import minimize_for_cloud, prepare_cloud_text, redact, truncate_middle

TRACEBACK = (
    "Traceback (most recent call last):\n"
    '  File "app.py", line 12, in <module>\n'
    "    main()\n"
    '  File "app.py", line 8, in main\n'
    "    total = sum(values)\n"
    "TypeError: unsupported operand type(s) for +: 'int' and 'str'"
)


# --- machine output is kept local ------------------------------------------


@pytest.mark.parametrize(
    ("text", "kind", "absent"),
    [
        (f"Why does this fail?\n{TRACEBACK}\nWhat should I change?", "Python traceback", "sum(values)"),
        ("Review this:\n```python\ndef f(x):\n    return x + 1\n```\nIs it right?", "code block", "return x + 1"),
        ("Unclosed fence:\n```\nsecret_value = 42", "code block", "secret_value"),
        ("My env:\nOPENAI_API_KEY=sk-a\nDB_HOST=h\nDB_PASS=p\nwhy?", "terminal output", "DB_HOST"),
        ("PS C:\\AI> dir\nPS C:\\AI> Get-Process\nPS C:\\AI> exit\nwhat happened?", "terminal output", "Get-Process"),
        ("$ ls\n$ cat notes.txt\n$ rm -rf build\nhelp", "terminal output", "notes.txt"),
        ("Logs:\n2026-09-21 10:00:01 start\n2026-09-21 10:00:02 load\n2026-09-21 10:00:03 fail", "terminal output", "load"),
        ('{"user": "sam", "token": "abc", "items": [1, 2]}', "JSON document", "sam"),
        ('Here is the body:\n{\n  "a": 1,\n  "b": [2', "JSON document", '"a"'),
        ("Page:\n<html><body><p>private</p></body></html>", "markup", "private"),
        ("key: -----BEGIN RSA PRIVATE KEY-----\nMIIEow\n-----END RSA PRIVATE KEY-----", "key material", "MIIEow"),
    ],
)
def test_machine_output_is_replaced_by_a_description(text: str, kind: str, absent: str) -> None:
    minimized, kept = minimize_for_cloud(text)

    assert absent not in minimized, "the pasted content itself must not survive"
    assert kind in kept
    assert "kept local" in minimized


def test_the_question_around_a_dump_survives() -> None:
    minimized, _ = minimize_for_cloud(f"Why does this fail?\n{TRACEBACK}\nWhat should I change?")

    assert minimized.startswith("Why does this fail?")
    assert minimized.endswith("What should I change?")
    assert "[Python traceback kept local: 6 lines]" in minimized


# --- prose is left alone ----------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Draft a short, polite email telling a supplier we are delaying the order by two weeks.",
        "Line one of a longer request.\nLine two.\nLine three.\nLine four.\nLine five.\n"
        "Line six.\nLine seven.\nLine eight.\nLine nine.\nLine ten.",
        "Open C:\\AI\\Config\\models.json and tell me which model is active.",
        "Is [1, 2, 3] a list or a tuple in Python?",
        "How do I make text bold with <b> in HTML?",
        "Explain the {curly braces} syntax in f-strings.",
        "Run `pytest -q` and tell me what fails.",
        "1. Install it\n2. Configure it\n3. Start it",
    ],
)
def test_prose_is_not_mistaken_for_a_dump(text: str) -> None:
    minimized, kept = minimize_for_cloud(text)

    assert minimized == text
    assert kept == []


def test_a_single_machine_looking_line_is_not_a_run() -> None:
    """One path or one assignment in a sentence is part of the question."""
    text = "Why does DB_HOST=localhost not resolve?"
    assert minimize_for_cloud(text) == (text, [])


# --- redaction and truncation ----------------------------------------------


def test_redaction_still_applies_to_what_remains() -> None:
    minimized, _ = minimize_for_cloud("my key is sk-abcdefghijklmnopqrstuvwx, is that bad?")
    prepared, kinds = prepare_cloud_text(minimized, max_chars=500)

    assert "sk-abcdefghijklmnopqrstuvwx" not in prepared
    assert "openai-key" in kinds


def test_redaction_reports_kinds_never_values() -> None:
    result = redact("Authorization: Bearer abcdefghijklmnopqrstuvwxyz")

    assert "abcdefghijklmnopqrstuvwxyz" not in result.text
    assert result.kinds and all("abcdef" not in kind for kind in result.kinds)


@pytest.mark.parametrize("limit", [0, 1, 20, 33, 60, 200])
def test_truncation_never_exceeds_its_limit(limit: int) -> None:
    assert len(truncate_middle("x" * 5000, limit)) <= limit


def test_truncation_keeps_both_ends() -> None:
    text = "START " + "filler " * 500 + " END"
    cut = truncate_middle(text, 200)

    assert cut.startswith("START")
    assert cut.endswith("END")
    assert "[... omitted ...]" in cut
