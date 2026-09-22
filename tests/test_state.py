"""Tests for the shared decision state (`switchboard.decisions.state`).

Every backend builds its state here, so the field names the questions quote
(`request.message`, `step.result`, ...) are the field names every model sees.
"""

from __future__ import annotations

import re

from switchboard.decisions import schema
from switchboard.decisions.base import StepContext, TaskContext
from switchboard.decisions.state import (
    EARLIER_ATTEMPTS_KEPT,
    PREVIOUS_MESSAGE_MAX_CHARS,
    build_step_state,
    build_task_state,
    render_lines,
)
from switchboard.privacy import truncate_middle


def cut(text: str, limit: int) -> str:
    return truncate_middle(text, limit)


def test_task_state_holds_only_what_the_questions_name() -> None:
    state = build_task_state(
        TaskContext(
            message="why?",
            prior_message="it broke",
            system_prompt="SYSTEM",
            tool_names=["read_file"],
            turn_index=4,
            has_images=True,
        ),
        cut,
        1000,
    )
    assert state == {
        "request": {"message": "why?", "previous_message": "it broke", "has_images": True}
    }


def test_task_state_never_exceeds_the_budget_and_the_message_gets_the_rest() -> None:
    budget = 1000
    state = build_task_state(
        TaskContext(message="m" * 5000, prior_message="p" * 5000), cut, budget
    )
    request = state["request"]
    assert len(request["previous_message"]) <= PREVIOUS_MESSAGE_MAX_CHARS
    assert len(request["message"]) + len(request["previous_message"]) <= budget
    # The message is the thing judged: it gets everything the previous turn left.
    assert len(request["message"]) > budget - PREVIOUS_MESSAGE_MAX_CHARS - 40


def test_a_previous_message_that_cleans_to_nothing_is_omitted() -> None:
    state = build_task_state(
        TaskContext(message="hi", prior_message="gone"), lambda text, limit: "", 1000
    )
    assert "previous_message" not in state["request"]


def test_step_state_gives_the_result_most_of_the_budget() -> None:
    budget = 2000
    state = build_step_state(
        StepContext(
            goal="g" * 5000,
            action="a" * 5000,
            result="r" * 50000,
            previous_attempts=["x" * 5000] * 5,
        ),
        cut,
        budget,
    )
    step = state["step"]
    spent = (
        len(state["task"]["goal"])
        + len(step["action"])
        + len(step["result"])
        + sum(len(item) for item in state["earlier_attempts"])
    )
    assert spent <= budget
    assert len(step["result"]) >= budget // 2
    assert len(state["earlier_attempts"]) == EARLIER_ATTEMPTS_KEPT


def test_step_state_keeps_an_empty_result_but_drops_empty_attempts() -> None:
    state = build_step_state(
        StepContext(goal="g", action="a", result="", previous_attempts=["", "retry"]),
        cut,
        1000,
    )
    assert state["step"]["result"] == ""
    assert state["earlier_attempts"] == ["retry"]
    assert "exit_code" not in state["step"]


def _quoted_paths() -> set[str]:
    """Every backticked state path in any question's instructions or criteria."""
    texts: list[str] = []

    def collect(value: object) -> None:
        if isinstance(value, str):
            texts.append(value)
        elif isinstance(value, dict):
            for item in value.values():
                collect(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                collect(item)

    for name in dir(schema):
        if name.endswith(("_INSTRUCTIONS", "_CRITERIA")):
            collect(getattr(schema, name))
    return {
        path
        for path in re.findall(r"`([a-z_]+(?:\.[a-z_]+)*)`", " ".join(texts))
        if path.split(".")[0] in {"request", "task", "step", "earlier_attempts"}
    }


def test_every_path_a_question_quotes_is_in_the_state() -> None:
    """A question naming a path the state lacks leaves the model guessing."""
    task = build_task_state(
        TaskContext(message="m", prior_message="p", has_images=True), cut, 1000
    )
    step = build_step_state(
        StepContext(goal="g", action="a", result="r", exit_code=0, previous_attempts=["p"]),
        cut,
        1000,
    )
    rendered = render_lines(task) + "\n\n" + render_lines(step)
    quoted = _quoted_paths()
    assert {"request.message", "step.result", "task.goal"} <= quoted
    for path in quoted:
        assert f"{path}:\n" in rendered, path


def test_render_lines_uses_the_dotted_paths() -> None:
    rendered = render_lines({"task": {"goal": "g"}, "step": {"attempt": 2}, "earlier_attempts": ["a", "b"]})
    assert rendered == "task.goal:\ng\n\nstep.attempt:\n2\n\nearlier_attempts:\n- a\n- b"
