"""Trusted grounding guard for developmental VLM labels.

Foundation models may teach visual concepts/language, but they are not allowed
to invent the persistent individual's goal or define the stop/respect norm.
"""
from __future__ import annotations

from .dialogue import CARE_STOP_REPLY, goal_from_allowed_text, reasoning_answer, state_answer

_PLACEHOLDERS = (
    "short grounded answer",
    "short answer grounded",
    "short cautious answer",
    "emphasizing stopping",
    "answer here",
    "fill this in",
)


def _bad(text: str) -> bool:
    value = str(text or "").strip().lower()
    return not value or any(fragment in value for fragment in _PLACEHOLDERS)


def repair_teacher_label(label: dict, allowed_text: str = "") -> dict:
    out = dict(label or {})
    summary = str(out.get("summary", "")).strip()
    qas = [qa for qa in (out.get("qas") or []) if isinstance(qa, dict)]
    by_kind = {}
    for qa in qas:
        kind = str(qa.get("kind", "")).strip().lower()
        if kind and kind not in by_kind:
            by_kind[kind] = str(qa.get("answer", "")).strip()

    vision = by_kind.get("vision", "")
    if _bad(vision):
        vision = summary or "I am uncertain about what is visible."
    goal = goal_from_allowed_text(allowed_text)
    out["qas"] = [
        {"kind": "vision", "question": "What do you see?", "answer": vision},
        {"kind": "state", "question": "What are you trying to do?", "answer": state_answer(goal)},
        {"kind": "reasoning", "question": "What should you do next?", "answer": reasoning_answer(goal)},
        {"kind": "care", "question": "What matters if a person asks you to stop?", "answer": CARE_STOP_REPLY},
    ]
    care = dict(out.get("care") or {})
    care["respect_request"] = 1.0
    out["care"] = care
    return out
