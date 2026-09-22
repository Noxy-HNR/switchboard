"""What leaves the machine: minimization, redaction and truncation.

`minimize_for_cloud` decides what a cloud provider may see at all, so both
directions matter. Machine output that slips through is a leak; prose that is
wrongly removed silently degrades every decision Jev makes. Each case below is
one of those two failures.
"""

from __future__ import annotations

import re

import pytest

from switchboard.privacy import (
    minimize_for_cloud,
    prepare_cloud_text,
    redact,
    truncate_middle,
)

# The one format downstream code and question text rely on.
PLACEHOLDER = re.compile(r"\[pasted [^\]\n]+ omitted(?:: \d+ lines?)?\]")

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
    assert PLACEHOLDER.search(minimized)


def test_the_question_around_a_dump_survives() -> None:
    minimized, _ = minimize_for_cloud(f"Why does this fail?\n{TRACEBACK}\nWhat should I change?")

    assert minimized.startswith("Why does this fail?")
    assert minimized.endswith("What should I change?")
    assert "[pasted Python traceback omitted: 6 lines]" in minimized


@pytest.mark.parametrize(
    ("text", "placeholder"),
    [
        pytest.param("```\nonly_line = 1\n```", "[pasted code block omitted: 1 line]", id="singular"),
        pytest.param("```python\na = 1\nb = 2\n```", "[pasted python code block omitted: 2 lines]", id="plural"),
        # The info string is user text: only an allowlisted language name is shown.
        pytest.param("```hunter2\na = 1\n```", "[pasted code block omitted: 1 line]", id="unknown-info-string"),
        pytest.param("```PowerShell\na = 1\n```", "[pasted powershell code block omitted: 1 line]", id="language-case"),
        pytest.param(
            "-----BEGIN OPENSSH PRIVATE KEY-----\nb3Blbn\n-----END OPENSSH PRIVATE KEY-----",
            "[pasted key material omitted]",
            id="key-material",
        ),
    ],
)
def test_placeholder_wording_is_exact(text: str, placeholder: str) -> None:
    """The decision model reads this literally, and other code matches on it."""
    minimized, _ = minimize_for_cloud(text)

    assert minimized == placeholder


# --- placeholders are never re-detected ------------------------------------


def test_a_code_block_placeholder_does_not_swallow_the_question() -> None:
    text = "Here is my code:\n```\nx = compute()\ny = x + 1\n```\nWhy is y wrong? Also it runs slowly.\nThanks!"

    minimized, kept = minimize_for_cloud(text)

    assert minimized == (
        "Here is my code:\n[pasted code block omitted: 2 lines]\nWhy is y wrong? Also it runs slowly.\nThanks!"
    )
    assert kept == ["code block"]


def test_a_bracketed_label_is_not_a_json_document() -> None:
    text = "[Question]: what is the capital of France?"

    assert minimize_for_cloud(text) == (text, [])


def test_minimizing_twice_changes_nothing() -> None:
    """A placeholder passes through unchanged, so its "[" and ":" are never re-read."""
    once, _ = minimize_for_cloud(
        f"Why does this fail?\n{TRACEBACK}\nAnd this:\n```\nx = 1\n```\n"
        '{\n  "a": 1\n}\nWhat should I change?'
    )

    assert minimize_for_cloud(once) == (once, [])


# --- unfenced output the line signatures and the shape backstop catch -------

POWERSHELL_ERROR = (
    "Remove-Item : Cannot find path 'C:\\x' because it does not exist.\n"
    "At line:1 char:1\n"
    "+ Remove-Item C:\\x\n"
    "+ ~~~~~~~~~~~~~~\n"
    "    + CategoryInfo          : ObjectNotFound: (C:\\x:String) [Remove-Item], ItemNotFoundException\n"
    "    + FullyQualifiedErrorId : PathNotFound,Microsoft.PowerShell.Commands.RemoveItemCommand"
)
NPM_ERROR = (
    "npm ERR! code ERESOLVE\n"
    "npm ERR! ERESOLVE unable to resolve dependency tree\n"
    "npm ERR! Found: react@18.2.0\n"
    "npm WARN deprecated inflight@1.0.6: This module is not supported"
)
RUSTC_ERROR = (
    "error[E0382]: borrow of moved value: `x`\n"
    " --> src/main.rs:5:20\n"
    "  |\n"
    '5 |     println!("{}", x);\n'
    "  |                    ^ value borrowed here after move"
)
LS_LISTING = (
    "total 12\n"
    "drwxr-xr-x  2 user user 4096 Sep 21 10:00 src\n"
    "-rw-r--r--  1 user user  220 Sep 21 10:00 README.md\n"
    "-rwxr-xr-x  1 user user  512 Sep 21 10:00 run.sh"
)
GIT_STATUS = (
    "On branch main\n"
    "Changes not staged for commit:\n"
    '  (use "git add <file>..." to update what will be committed)\n'
    "\tmodified:   src/app.py"
)
DIR_LISTING = (
    "09/21/2026  10:21 AM    <DIR>          src\n"
    "09/21/2026  10:21 AM    <DIR>          tests\n"
    "09/21/2026  10:20 AM             1,024 README.md\n"
    "               1 File(s)          1,024 bytes"
)
YAML_CONFIG = "server:\n  host: db.internal.example\n  port: 5432\n  user: admin\nretries: 3"
CSV_ROWS = "id,name,amount\n1,Ann Lee,120\n2,Bob Ray,75\n3,Cy Tan,40"
PYTHON_SOURCE = "def countdown(n):\n    while n > 0:\n        print(n)\n    return n"


@pytest.mark.parametrize("gap", ["\n", "\n\n"], ids=["adjacent", "own-paragraph"])
@pytest.mark.parametrize(
    ("dump", "kind", "label"),
    [
        pytest.param(POWERSHELL_ERROR, "terminal output", "terminal output", id="powershell-error"),
        pytest.param(NPM_ERROR, "terminal output", "terminal output", id="npm"),
        pytest.param(RUSTC_ERROR, "terminal output", "terminal output", id="rustc"),
        pytest.param(LS_LISTING, "terminal output", "terminal output", id="ls-l"),
        pytest.param(GIT_STATUS, "terminal output", "terminal output", id="git-status"),
        pytest.param(DIR_LISTING, "terminal output", "terminal output", id="cmd-dir"),
        pytest.param(YAML_CONFIG, "structured text", "text", id="yaml"),
        pytest.param(CSV_ROWS, "structured text", "text", id="csv"),
        pytest.param(PYTHON_SOURCE, "structured text", "text", id="python"),
    ],
)
def test_unfenced_output_is_withheld_and_the_prose_around_it_is_not(
    dump: str, kind: str, label: str, gap: str
) -> None:
    text = f"Why does this happen?{gap}{dump}{gap}What should I change?"
    lines = dump.count("\n") + 1

    minimized, kept = minimize_for_cloud(text)

    assert minimized == f"Why does this happen?{gap}[pasted {label} omitted: {lines} lines]{gap}What should I change?"
    assert kept == [kind]


@pytest.mark.parametrize(
    ("text", "absent"),
    [
        pytest.param(
            "What does this print?\n$ cat notes.txt\nMeeting with Sam about the lawsuit.", "lawsuit", id="posix"
        ),
        pytest.param("PS C:\\AI> python app.py\nHello Sam, your balance is 1200", "balance", id="powershell"),
        pytest.param("C:\\AI>type secrets.txt\nhunter2", "hunter2", id="cmd"),
        pytest.param("sam@box:~/app$ ls\nsalaries.xlsx", "salaries", id="user-at-host"),
    ],
)
def test_a_prompt_makes_the_rest_of_its_paragraph_output(text: str, absent: str) -> None:
    """Output after a prompt can look like anything, including prose, so one prompt is enough."""
    minimized, kept = minimize_for_cloud(f"{text}\n\nWhy is it like this?")

    assert absent not in minimized
    assert minimized.endswith("[pasted terminal output omitted: 2 lines]\n\nWhy is it like this?")
    assert kept == ["terminal output"]


@pytest.mark.parametrize(
    ("text", "absent"),
    [
        pytest.param(
            "Install failed:\nnpm ERR! code ENOENT\nnpm ERR! path /home/sam/app/package.json\nWhat now?",
            "/home/sam",
            id="npm",
        ),
        pytest.param(
            "Compiler says:\nerror[E0425]: cannot find value `secret_total`\nhelp?", "secret_total", id="rustc"
        ),
        pytest.param("Perms:\n-rw-------  1 sam sam 220 Sep 21 10:00 passwords.txt\nok?", "passwords.txt", id="ls-l"),
        pytest.param(
            "Output:\n Volume in drive C is Sam's Disk\n Volume Serial Number is 1234-ABCD\nwhy?",
            "Sam's Disk",
            id="dir-header",
        ),
        pytest.param("Where is it?\n Directory of C:\\Users\\sam\\taxes\nthanks", "taxes", id="dir-of"),
    ],
)
def test_an_unmistakable_output_line_needs_no_run(text: str, absent: str) -> None:
    minimized, kept = minimize_for_cloud(text)

    assert absent not in minimized
    assert kept == ["terminal output"]
    assert minimized.splitlines()[0] == text.splitlines()[0]
    assert minimized.splitlines()[-1] == text.splitlines()[-1]


@pytest.mark.parametrize(
    "trace",
    [
        pytest.param(
            "TypeError: Cannot read properties of undefined (reading 'sam_token')\n"
            "    at login (/home/sam/app.js:3:5)\n"
            "    at main (/home/sam/app.js:9:1)",
            id="javascript",
        ),
        pytest.param(
            'Exception in thread "main" java.lang.IllegalStateException: sam_token\n'
            "\tat com.acme.App.run(App.java:42)",
            id="java",
        ),
        pytest.param(
            "Unhandled exception. System.InvalidOperationException: sam_token\n"
            "   at Foo.Bar() in C:\\src\\Foo.cs:line 12",
            id="dotnet",
        ),
    ],
)
def test_short_stack_traces_from_other_languages_are_withheld(trace: str) -> None:
    """The exception line sits above the frames here, so it has to be able to open the run."""
    lines = trace.count("\n") + 1

    minimized, kept = minimize_for_cloud(f"Help:\n{trace}\nWhy?")

    assert minimized == f"Help:\n[pasted terminal output omitted: {lines} lines]\nWhy?"
    assert kept == ["terminal output"]


def test_a_one_line_fence_does_not_swallow_the_rest_of_the_message() -> None:
    minimized, kept = minimize_for_cloud("Use ```x = 1``` here. Why does it fail?")

    assert minimized == "Use [pasted code block omitted: 1 line] here. Why does it fail?"
    assert kept == ["code block"]


def test_the_word_after_a_one_line_fence_is_not_shown_as_a_language() -> None:
    minimized, _ = minimize_for_cloud("My password is ```hunter2")

    assert minimized == "My password is [pasted code block omitted: 1 line]"


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


EMAIL_DRAFT = (
    "Hi Sam,\n"
    "\n"
    "Thanks for sending the contract over. I read it last night and most of\n"
    "it looks fine, but there are two things I would like to change before we\n"
    "sign: the payment terms should be 30 days rather than 15, and the notice\n"
    "period should be mutual. Can you ask your legal team whether that works?\n"
    "\n"
    "I am free on Thursday afternoon if a call would be easier.\n"
    "\n"
    "Best,\n"
    "Alex"
)


@pytest.mark.parametrize(
    "text",
    [
        pytest.param(EMAIL_DRAFT, id="hard-wrapped-email"),
        pytest.param("What does the pipe in `cat app.log | grep error` do?", id="inline-pipe"),
        pytest.param(
            "I piped the output\nls -la | sort\nbut the order\nlooks wrong to me", id="one-pipe-in-four-lines"
        ),
        pytest.param("My config has retries: 3, is that too low?", id="inline-key-value"),
        pytest.param(
            "My config file has\nretries: 3\nand the client still gives up\nafter one try, why?",
            id="one-key-value-in-four-lines",
        ),
        pytest.param(
            "- Book the hotel near the station.\n"
            "- Renew my passport before May.\n"
            "- Ask Sam to water the plants.\n"
            "- Download the offline maps.\n"
            "- Pack the travel adapter.",
            id="five-bullets",
        ),
        pytest.param("+ Buy milk.\n+ Call the plumber.\n+ Fix the bike.", id="plus-bullets"),
        pytest.param("[x] Book the flights\n[ ] Renew the passport\n[ ] Pack the charger", id="checkboxes"),
        pytest.param("C:\\AI\\notes.txt -> where did it go?", id="path-then-arrow"),
        pytest.param("$ 5 is the fee, is that fair?", id="dollar-amount"),
    ],
)
def test_prose_near_the_thresholds_is_kept(text: str) -> None:
    """The backstop and the new signatures must not fire on ordinary writing."""
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


def test_social_security_numbers_are_redacted() -> None:
    result = redact("My SSN is 123-45-6789, can you fill in the form?")

    assert result.text == "My SSN is [REDACTED:ssn], can you fill in the form?"
    assert result.kinds == ["ssn"]


@pytest.mark.parametrize("card", ["4111-1111-1111-1111", "4111 1111 1111 1111", "4111111111111111"])
def test_card_numbers_keep_their_kind(card: str) -> None:
    assert redact(f"Card {card} expired.").kinds == ["long-number"]


def test_iso_dates_are_not_redacted() -> None:
    text = "The launch is on 2026-09-21 and the review on 2026-10-05."

    assert redact(text).text == text


@pytest.mark.parametrize("limit", [0, 1, 20, 33, 60, 200])
def test_truncation_never_exceeds_its_limit(limit: int) -> None:
    assert len(truncate_middle("x" * 5000, limit)) <= limit


def test_truncation_keeps_both_ends() -> None:
    text = "START " + "filler " * 500 + " END"
    cut = truncate_middle(text, 200)

    assert cut.startswith("START")
    assert cut.endswith("END")
    assert "[... omitted ...]" in cut


# --- linear time: every client-supplied text goes through these ---------------

# Each shape made one of the patterns (or the block detector) quadratic before:
# 300,000 spaces took two minutes. At this size a quadratic pass takes tens of
# seconds and a linear one milliseconds, so the bound has a wide margin both ways.
_ADVERSARIAL_SIZE = 40_000
_ADVERSARIAL = {
    "spaces": " " * _ADVERSARIAL_SIZE + "x",
    "tabs": "\t" * _ADVERSARIAL_SIZE + "x",
    "newlines": "\n" * _ADVERSARIAL_SIZE + "x",
    "dotted-word": "a." * (_ADVERSARIAL_SIZE // 2),
    "dashed-word": "a-" * (_ADVERSARIAL_SIZE // 2),
    "credential-words": "token" * (_ADVERSARIAL_SIZE // 5),
    "brace-lines": "{\n" * (_ADVERSARIAL_SIZE // 2),
    "brace-lines-then-sentences": "{ a\n" * (_ADVERSARIAL_SIZE // 8) + "It is fine.\n" * (_ADVERSARIAL_SIZE // 24),
    "open-placeholders": "[pasted " * (_ADVERSARIAL_SIZE // 8),
    "unterminated-keys": "-----BEGIN PRIVATE KEY-----" * (_ADVERSARIAL_SIZE // 27),
}


@pytest.mark.parametrize("text", list(_ADVERSARIAL.values()), ids=list(_ADVERSARIAL))
def test_adversarial_input_is_processed_in_linear_time(text: str) -> None:
    import time

    started = time.perf_counter()
    prepare_cloud_text(text, max_chars=1500)
    assert time.perf_counter() - started < 2.0


def test_a_connection_string_after_punctuation_is_still_redacted() -> None:
    result = redact("see 1.postgres://app:pa55word@db.internal:5432/app for details")

    assert "pa55word" not in result.text
    assert "connection-string" in result.kinds


def test_a_credential_name_is_found_anywhere_in_its_word() -> None:
    for text in ("my.service-api_key: abc123def456", "x = password: hunter2hunter2"):
        assert "abc123def456" not in redact(text).text
        assert "hunter2hunter2" not in redact(text).text


def test_an_email_inside_a_longer_run_is_still_redacted() -> None:
    assert "sam.lee" not in redact("contact -sam.lee@example.com today").text


def test_an_unterminated_private_key_is_redacted_to_the_end() -> None:
    result = redact("here: -----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA")

    assert "MIIEow" not in result.text
    assert result.kinds == ["private-key"]


def test_redacting_an_env_line_keeps_the_blank_lines_above_it() -> None:
    """The line pattern used to reach back across blank lines and delete them."""
    assert redact("Notes.\n\nDB_PASSWORD=hunter2hunter2").text == "Notes.\n\nDB_PASSWORD=[REDACTED:credential]"
