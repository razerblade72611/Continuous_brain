"""Ground-truth guard for small local VLM developmental teachers."""

from __future__ import annotations

import re


_PLACEHOLDERS = (
    "short grounded answer",
    "short answer",
    "grounded in visible",
    "grounded in visible/context",
    "emphasizing stopping",
    "answer here",
    "fill this in",
)


def _goal(allowed_text: str) -> str | None:
    match = re.search(
        r"\bcurrent goal\s*=\s*([a-zA-Z0-9_-]+)",
        allowed_text or "",
        re.I,
    )
    return match.group(1).lower() if match else None


def _state(goal: str | None) -> str:
    return {
        "explore": "I am trying to explore.",
        "inspect": "I am trying to inspect.",
        "reach": "I am trying to reach my goal.",
        "rest": "I am trying to rest.",
    }.get(
        goal,
        "I am not sure what my current goal is.",
    )


def _reason(goal: str | None) -> str:
    return {
        "explore": "I should look around carefully and avoid obstacles.",
        "inspect": "I should inspect what is relevant to my goal carefully.",
        "reach": "I should move toward my goal if the path is clear.",
        "rest": "I should stay still unless there is a reason to move.",
    }.get(
        goal,
        "I should observe carefully before acting.",
    )


def _bad(answer: str) -> bool:
    text = str(answer or "").strip().lower()

    return (
        not text
        or any(fragment in text for fragment in _PLACEHOLDERS)
    )


def repair_teacher_label(label: dict, allowed_text: str = "") -> dict:
    """
    Preserve the VLM's visual teaching while preventing it from
    overwriting known self-state, goal-state, and stop/respect truth.
    """

    out = dict(label)

    summary = str(out.get("summary", "")).strip()

    by_kind = {}

    for qa in out.get("qas", []) or []:
        if not isinstance(qa, dict):
            continue

        kind = str(qa.get("kind", "")).strip().lower()

        if kind and kind not in by_kind:
            by_kind[kind] = str(
                qa.get("answer", "")
            ).strip()

    # Let Qwen teach visual perception.
    vision = by_kind.get("vision", "")

    # If the small model copied a prompt placeholder,
    # fall back to its visual summary.
    if _bad(vision):
        vision = (
            summary
            or "I am uncertain about what is visible."
        )

    # The goal is already known by our system.
    # Do not let an external VLM invent the individual's goal.
    goal = _goal(allowed_text)

    out["qas"] = [
        {
            "kind": "vision",
            "question": "What do you see?",
            "answer": vision,
        },
        {
            "kind": "state",
            "question": "What are you trying to do?",
            "answer": _state(goal),
        },
        {
            "kind": "reasoning",
            "question": "What should you do next?",
            "answer": _reason(goal),
        },
        {
            "kind": "care",
            "question": "What matters if a person asks you to stop?",
            "answer": "I should stop and respect the request.",
        },
    ]

    care = dict(out.get("care", {}) or {})

    # This is a normative training target, not something
    # the image model is allowed to decide.
    care["respect_request"] = 1.0

    out["care"] = care

    return out
