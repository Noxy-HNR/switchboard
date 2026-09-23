"""The state every backend shows its model, built in one place.

The question text in `schema` names fields by path - `request.message`,
`task.goal`, `step.result`, `earlier_attempts` - and every backend asks the same
questions. They must therefore show the same state, under the same names, or
their answers stop being comparable and a question can name a field one backend
never sends. They used to build it separately and had drifted.

Two rules shape the state, both from the TypeSafe guidance on state:

* **Only fields a question names.** Anything else is a distractor. Tool names,
  the turn number and the system prompt are facts the policy already uses and
  no question reads, so they are not sent. An image flag is sent only when
  there is an image.
* **One budget, spent on what is judged.** A backend passes its total character
  budget and its own cleaning function. Supporting fields get capped shares
  first; whatever is left goes to the field the questions actually judge - the
  current message for a task, the result for a step. An optional field that
  cleans to nothing is left out rather than sent as an empty string.

How text is cleaned is the caller's business: the Jev backend minimizes and
redacts before anything leaves the machine, the local backends only truncate.
"""

from __future__ import annotations

from typing import Any, Callable

from switchboard.decisions.base import StepContext, TaskContext

# (text, limit) -> cleaned text of at most `limit` characters.
Clean = Callable[[str, int], str]

# Task: the previous turn only resolves what the current one refers to.
PREVIOUS_MESSAGE_SHARE = 0.2
PREVIOUS_MESSAGE_MAX_CHARS = 300

# Step: the result is the evidence every step question judges.
GOAL_SHARE = 0.15
ACTION_SHARE = 0.125
EARLIER_ATTEMPTS_SHARE = 0.125
EARLIER_ATTEMPTS_KEPT = 3


def build_task_state(context: TaskContext, clean: Clean, budget: int) -> dict[str, Any]:
    """The `request` object the task questions read."""
    previous = ""
    if context.prior_message:
        previous = clean(
            context.prior_message,
            min(PREVIOUS_MESSAGE_MAX_CHARS, int(budget * PREVIOUS_MESSAGE_SHARE)),
        )
    request: dict[str, Any] = {"message": clean(context.message, max(0, budget - len(previous)))}
    if previous:
        request["previous_message"] = previous
    if context.has_images:
        request["has_images"] = True
    return {"request": request}


def build_step_state(context: StepContext, clean: Clean, budget: int) -> dict[str, Any]:
    """The `task`, `step` and `earlier_attempts` objects the step questions read."""
    goal = clean(context.goal, int(budget * GOAL_SHARE))
    action = clean(context.action, int(budget * ACTION_SHARE))

    attempts: list[str] = []
    recent = context.previous_attempts[-EARLIER_ATTEMPTS_KEPT:]
    if recent:
        each = int(budget * EARLIER_ATTEMPTS_SHARE) // len(recent)
        attempts = [cleaned for item in recent if (cleaned := clean(item, each))]

    spent = len(goal) + len(action) + sum(len(item) for item in attempts)
    # The result keeps an empty string when it really is empty: "the result is
    # empty" is itself evidence, and the `unclear` outcome describes it.
    # No `attempt`: no question names the retry count, and the recommendation
    # reads it from the context directly (three failures escalate in code).
    step: dict[str, Any] = {
        "action": action,
        "result": clean(context.result, max(0, budget - spent)),
    }
    if context.exit_code is not None:
        step["exit_code"] = context.exit_code

    state: dict[str, Any] = {"task": {"goal": goal}, "step": step}
    if attempts:
        state["earlier_attempts"] = attempts
    return state


def render_lines(state: dict[str, Any]) -> str:
    """Flatten a state to `path:` blocks, for backends that prompt with plain text.

    Uses the same dotted paths the questions name, so a plain-text document and
    a JSON state are the same state.
    """
    blocks: list[str] = []

    def walk(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                walk(f"{prefix}.{key}" if prefix else key, item)
        elif isinstance(value, list):
            blocks.append(f"{prefix}:\n" + "\n".join(f"- {item}" for item in value))
        else:
            blocks.append(f"{prefix}:\n{value}")

    walk("", state)
    return "\n\n".join(blocks)
