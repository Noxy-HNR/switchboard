"""The decision surface: what Switchboard asks, and what an answer looks like.

Two judgments exist, and every backend must answer both:

* a **task** judgment about an incoming request (what kind of work is this, how
  hard, does it need tools) used to pick a model;
* a **step** judgment about a finished agent step (did it work, is the task
  done) used to advise retry/escalate/stop.

Deliberately absent: anything code can know exactly. Context length, model
residency, VRAM, endpoint health and switch cost are facts, not judgments, and
live in `registry`/`upstream`. The policy combines the two.

The question text below is shared by every backend so results stay comparable:
Jev, a local model, and the rule engine all answer the same questions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Literal

# --- Task judgments ---------------------------------------------------------

Domain = Literal["general", "coding", "medical", "writing", "research", "system_ops"]

DOMAINS: tuple[Domain, ...] = ("general", "coding", "medical", "writing", "research", "system_ops")

# Criteria are written so each option stands alone, per the Jev guidance on
# literal reading: the model answers the words, so boundary cases go here -
# including which neighbouring option a borderline case belongs to.
DOMAIN_CRITERIA: dict[str, str] = {
    "general": (
        "Everyday conversation, general knowledge, advice or explanation that needs "
        "no specialist expertise. Explaining or summarising code, health material or "
        "this computer's setup belongs to that option instead. Also anything that fits "
        "none of the other options."
    ),
    "coding": (
        "Writing, reading, debugging, reviewing or explaining source code, scripts, "
        "configuration files or APIs, including a script or command for the user to "
        "run themselves."
    ),
    "medical": (
        "Health, clinical or biomedical content where medical expertise changes the "
        "quality of the answer: symptoms, diagnoses, treatments, drugs and doses, lab "
        "results, medical literature, including explaining or summarising them. Not a "
        "passing mention of health inside some other request."
    ),
    "writing": (
        "Drafting or editing prose for a human audience where wording, tone and "
        "structure are the work: emails, essays, stories, documentation wording. Not "
        "prose whose correctness depends on medical or coding knowledge."
    ),
    "research": (
        "Gathering, comparing or synthesising information from several sources to "
        "reach a conclusion, such as a literature review or a product comparison. Not "
        "a single factual question."
    ),
    "system_ops": (
        "Having the assistant itself operate this Windows computer: files and folders, "
        "running terminal commands, processes, services, installed software, scheduled "
        "tasks, automation. Not writing a script for the user to run (that is coding)."
    ),
}

DOMAIN_INSTRUCTIONS = (
    "Which kind of expertise does a good answer to `request.message` depend on most? "
    "Judge what the user wants done now; use `request.previous_message`, when present, "
    "only to resolve what the message refers to. A topic mentioned only in passing "
    "does not count."
)

class Difficulty(IntEnum):
    """Ordered levels for the difficulty Score. Index matches the Score level."""

    TRIVIAL = 0
    ROUTINE = 1
    MODERATE = 2
    HARD = 3


# Each level describes a situation rather than a degree, and the question asks
# about one dimension only - reasoning - because domain expertise is already
# carried by the domain answer and the model's capability ceiling for it.
DIFFICULTY_CRITERIA: list[str] = [
    # Level 0
    "A greeting, thanks or acknowledgement, or a question answered in one sentence "
    "from common knowledge, such as a definition or a unit conversion.",
    # Level 1
    "One clear task done in a single pass without planning: a short explanation, a "
    "small edit to supplied text or code, a simple script, or a summary of a short "
    "passage.",
    # Level 2
    "Several dependent steps that each have to be right: debugging a real error from "
    "its symptoms, designing a small component, or solving a multi-part problem.",
    # Level 3
    "A long chain of reasoning where an early mistake spoils everything after it, or "
    "weighing several competing trade-offs to reach one recommendation: designing a "
    "system, writing a proof, or planning a migration under many constraints.",
]

# Pasted machine output is replaced by "[pasted <kind> omitted: N lines]" before
# anything is sent to a cloud provider (see privacy.minimize_for_cloud). Read
# literally, such a marker looks like missing evidence, so the questions it can
# affect say what it means.
_MARKER_NOTE = (
    "A `[pasted ... omitted]` marker stands for content the assistant will see in full."
)

DIFFICULTY_INSTRUCTIONS = (
    "How much reasoning does a good answer to `request.message` take? Judge the "
    "thinking the answer needs, not the length of the message or how specialised its "
    f"topic is. {_MARKER_NOTE}"
)

TOOL_NEED_INSTRUCTIONS = (
    "Answering `request.message` correctly requires taking an action outside the "
    "conversation, such as reading or writing files, running a command, controlling "
    "this computer, or fetching a web page."
)
TOOL_NEED_CRITERIA = {
    "true": "The answer depends on something only an action can supply: a file or "
    "program state on this computer that was not pasted, the result of running "
    "something, or information newer than general knowledge.",
    "false": "The request can be answered from general knowledge and the conversation "
    "alone, including when it asks for code or commands for the user to run.",
}

AMBIGUITY_INSTRUCTIONS = (
    "Even using `request.previous_message` for context, `request.message` is too "
    "unclear to act on without first asking the user what they mean."
)
AMBIGUITY_CRITERIA = {
    "true": "A competent assistant would have to ask before starting: the message "
    "refers to something absent from the conversation, or its goal could mean clearly "
    "different things.",
    "false": "A competent assistant could start work, assuming reasonable defaults for "
    "missing details. An attached image (`request.has_images`) or a `[pasted ... "
    "omitted]` marker counts as content the assistant can see.",
}

@dataclass(slots=True)
class TaskDecision:
    """The semantic half of a routing decision."""

    domain: Domain = "general"
    domain_probabilities: dict[str, float] = field(default_factory=dict)
    domain_confidence: float | None = None
    # Probability-weighted position on the Difficulty scale; can land between levels.
    difficulty: float = 1.0
    difficulty_probabilities: dict[str, float] = field(default_factory=dict)
    difficulty_confidence: float | None = None
    # Probability that the answer is yes. Noul answers carry no confidence.
    tool_need: float = 0.0
    ambiguity: float = 0.0

    backend: str = "rules"
    latency_ms: float = 0.0
    # Whether the probability dicts are calibrated probabilities (Jev) rather than
    # heuristic scores (rules) or a next-token distribution (logit). Only
    # calibrated distributions may be used as weights; everything else is read
    # as a point value.
    calibrated: bool = False
    # True when the backend answered only some questions and defaults filled the rest.
    degraded: bool = False
    usage: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def difficulty_level(self) -> Difficulty:
        return Difficulty(max(0, min(3, round(self.difficulty))))

    def summary(self) -> dict[str, Any]:
        """Compact, storable form. No raw request text ever enters this."""
        return {
            "domain": self.domain,
            "domain_confidence": _round(self.domain_confidence),
            "domain_probabilities": {k: _round(v) for k, v in self.domain_probabilities.items()},
            "difficulty": _round(self.difficulty),
            "difficulty_confidence": _round(self.difficulty_confidence),
            "tool_need": _round(self.tool_need),
            "ambiguity": _round(self.ambiguity),
            "backend": self.backend,
            "latency_ms": _round(self.latency_ms, 1),
            "calibrated": self.calibrated,
            "degraded": self.degraded,
            "notes": self.notes,
        }


# --- Step judgments ---------------------------------------------------------

StepOutcome = Literal["success", "partial", "failure", "unclear"]

STEP_OUTCOMES: tuple[StepOutcome, ...] = ("success", "partial", "failure", "unclear")

STEP_OUTCOME_CRITERIA: dict[str, str] = {
    "success": "The step did what it set out to do; its result is usable as-is.",
    "partial": "The step did some of what it set out to do, but part of it did not "
    "work or is missing.",
    "failure": "The step did not do what it set out to do: it errored, was rejected, "
    "or produced an unusable result.",
    "unclear": "`step.result` does not show whether the action worked: it is empty, "
    "cut off, or about something else.",
}

STEP_OUTCOME_INSTRUCTIONS = (
    "Judging from `step.result` and `step.exit_code` (when present): did "
    "`step.action` do what it set out to do?"
)

STEP_PROGRESS_INSTRUCTIONS = (
    "After this step, something needed for `task.goal` is known or done that was not "
    "known or done before, including by `earlier_attempts` when present."
)
STEP_PROGRESS_CRITERIA = {
    "true": "Something needed for the goal is now known or done that was not before.",
    "false": "The step repeated earlier work, went in circles, or left the task where it was.",
}

STEP_COMPLETE_INSTRUCTIONS = (
    "`step.result` shows that everything `task.goal` asks for is now done, with no "
    "further step needed."
)
STEP_COMPLETE_CRITERIA = {
    "true": "The result itself shows the whole goal is satisfied, such as the final "
    "check passing or the requested output existing.",
    "false": "The result shows only part of the goal done, or does not show whether "
    "the rest is done.",
}

# Asked about every step, so the premise is stated: it only means anything when
# the action failed. A typo is a mistake in the action, so an identical re-run
# fails again - it belongs under "true", not with transient causes.
STEP_NEW_APPROACH_INSTRUCTIONS = (
    "If `step.action` failed: running it again exactly as written would fail again "
    "for the same reason."
)
STEP_NEW_APPROACH_CRITERIA = {
    "true": "The cause shown in `step.result` would recur on an identical re-run: a "
    "missing capability or dependency, the wrong tool, a rejected permission, or a "
    "mistake in the action itself such as a typo or a wrong path.",
    "false": "The cause is transient and may clear on a plain retry: a timeout, a "
    "dropped connection, a busy resource, or a temporary lock.",
}

# Recommendations are decided in code from the judgments plus hard facts.
Recommendation = Literal[
    "continue", "retry", "retry_different", "escalate", "ask_user", "stop_complete", "stop_failed"
]


@dataclass(slots=True)
class StepDecision:
    """The semantic half of an agent-step evaluation."""

    outcome: StepOutcome = "unclear"
    outcome_probabilities: dict[str, float] = field(default_factory=dict)
    outcome_confidence: float | None = None
    progress: float = 0.5
    complete: float = 0.0
    needs_new_approach: float = 0.5

    backend: str = "rules"
    latency_ms: float = 0.0
    # See TaskDecision.calibrated: only Jev's answers are calibrated.
    calibrated: bool = False
    degraded: bool = False
    usage: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome,
            "outcome_confidence": _round(self.outcome_confidence),
            "outcome_probabilities": {
                k: _round(v) for k, v in self.outcome_probabilities.items()
            },
            "progress": _round(self.progress),
            "complete": _round(self.complete),
            "needs_new_approach": _round(self.needs_new_approach),
            "backend": self.backend,
            "latency_ms": _round(self.latency_ms, 1),
            "calibrated": self.calibrated,
            "degraded": self.degraded,
            "notes": self.notes,
        }


def _round(value: float | None, digits: int = 3) -> float | None:
    return None if value is None else round(float(value), digits)


def score_from_probabilities(probabilities: dict[str, float], levels: int) -> float:
    """Probability-weighted level, matching how a Score answer is defined."""
    total = 0.0
    weight = 0.0
    for key, probability in probabilities.items():
        try:
            level = int(key)
        except (TypeError, ValueError):
            continue
        if 0 <= level < levels:
            total += level * probability
            weight += probability
    return total / weight if weight else 0.0
