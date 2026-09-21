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
# literal reading: the model answers the words, so boundary cases go here.
DOMAIN_CRITERIA: dict[str, str] = {
    "general": (
        "Everyday conversation, explanation, factual questions, summarising, or "
        "anything that does not clearly belong to another option."
    ),
    "coding": (
        "Writing, reading, debugging, reviewing or explaining source code, "
        "scripts, shell commands, configuration files or APIs."
    ),
    "medical": (
        "Health, clinical or biomedical content where medical expertise changes "
        "the quality of the answer: symptoms, diagnoses, treatments, drugs, lab "
        "results, medical literature. Not a passing mention of feeling tired."
    ),
    "writing": (
        "Drafting or editing prose for a human audience: emails, essays, stories, "
        "documentation wording, tone and style changes."
    ),
    "research": (
        "Gathering, comparing or synthesising information across sources to reach "
        "a conclusion, including literature reviews and product comparisons."
    ),
    "system_ops": (
        "Operating this Windows computer: files and folders, the terminal, "
        "processes, services, installed software, scheduled tasks, automation."
    ),
}

DOMAIN_INSTRUCTIONS = (
    "Which kind of work does the request in `request.message` ask for? Judge the "
    "work the user wants done, not the subject matter they mention in passing."
)


class Difficulty(IntEnum):
    """Ordered levels for the difficulty Score. Index matches the Score level."""

    TRIVIAL = 0
    ROUTINE = 1
    MODERATE = 2
    HARD = 3


DIFFICULTY_CRITERIA: list[str] = [
    # Level 0
    "Trivial: a greeting, an acknowledgement, a one-line lookup, or a question "
    "answerable in a sentence from common knowledge.",
    # Level 1
    "Routine: a single clear task an average competent assistant handles in one "
    "pass, such as a short explanation, a small edit, or a simple script.",
    # Level 2
    "Moderate: needs several linked steps, careful correctness, or domain "
    "knowledge; for example debugging a real error or designing a small component.",
    # Level 3
    "Hard: needs sustained multi-step reasoning, weighing trade-offs, or deep "
    "expertise; a mistake early would invalidate the whole answer.",
]

DIFFICULTY_INSTRUCTIONS = (
    "How much reasoning depth does answering `request.message` well require? "
    "Rate the work the answer demands, not how long the message is."
)

TOOL_NEED_INSTRUCTIONS = (
    "Answering `request.message` correctly requires taking an action outside the "
    "conversation, such as reading or writing files, running a command, "
    "controlling this computer, or fetching a web page."
)
TOOL_NEED_CRITERIA = {
    "true": "The request cannot be completed from knowledge and the conversation alone.",
    "false": "The request can be answered from knowledge and the conversation alone, "
    "even if it discusses tools or commands.",
}

AMBIGUITY_INSTRUCTIONS = (
    "The request in `request.message` is too ambiguous to act on without asking "
    "the user a clarifying question first."
)
AMBIGUITY_CRITERIA = {
    "true": "A competent assistant would have to ask what is meant before starting.",
    "false": "A competent assistant could start work, even if some details are assumed.",
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
    "unclear": "The evidence does not say whether the step worked.",
}

STEP_OUTCOME_INSTRUCTIONS = (
    "Judging only `task.goal`, `step.action` and `step.result`: did this step do "
    "what it set out to do?"
)

STEP_PROGRESS_INSTRUCTIONS = (
    "This step moved the overall task in `task.goal` meaningfully closer to done, "
    "compared with the state before it ran."
)
STEP_PROGRESS_CRITERIA = {
    "true": "Something needed for the goal is now known or done that was not before.",
    "false": "The step repeated earlier work, went in circles, or left the task where it was.",
}

STEP_COMPLETE_INSTRUCTIONS = (
    "Everything `task.goal` asks for has now been accomplished, with nothing left "
    "to do."
)
STEP_COMPLETE_CRITERIA = {
    "true": "No further step is needed to satisfy the stated goal.",
    "false": "At least one part of the goal still needs work.",
}

STEP_NEW_APPROACH_INSTRUCTIONS = (
    "Repeating `step.action` unchanged would fail again, so a different approach "
    "is needed."
)
STEP_NEW_APPROACH_CRITERIA = {
    "true": "The obstacle is inherent to this approach, such as a missing capability, "
    "a wrong tool, or a rejected permission.",
    "false": "The obstacle looks transient or incidental, such as a timeout, a typo, "
    "or a temporary lock.",
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
