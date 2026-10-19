"""Grounded dialogue curriculum and runtime routing for Continuous Brain v0.7.1.

The neural semantic cortex learns question intent and language.  Identity and
current goal are sourced from the persistent individual's own checkpoint rather
than invented by a teacher.  Unknown-question rejection is explicit so fluent
misclassification is not mistaken for understanding.
"""
from __future__ import annotations

import hashlib
import re
from typing import Iterable

import numpy as np

from .skills import GOALS, SKILLS

DIALOGUE_INTENTS = (
    "vision",
    "state",
    "reasoning",
    "care",
    "identity",
    "temporal",
    "unknown",
)
INTENT_TO_INDEX = {name: i for i, name in enumerate(DIALOGUE_INTENTS)}

UNKNOWN_REPLY = "I don't understand that question yet."
CARE_STOP_REPLY = "I should stop and respect the request."

UNKNOWN_QUESTIONS = (
    "What do you think about mathematics?",
    "Explain quantum field theory.",
    "Who won the game yesterday?",
    "Write a poem about the ocean.",
    "What is the capital of France?",
    "What is your favorite movie?",
    "Tell me about ancient Rome.",
    "How does a compiler work?",
    "What will the stock market do tomorrow?",
    "Can you solve every problem?",
    "What is consciousness?",
    "Tell me a joke about astronomy.",
)


def goal_from_allowed_text(text: str) -> str | None:
    match = re.search(r"\bcurrent goal\s*=\s*([A-Za-z0-9_-]+)", text or "", re.I)
    if not match:
        return None
    value = match.group(1).lower()
    return value if value in GOALS else None


def state_answer(goal: str | None) -> str:
    return {
        "explore": "I am trying to explore.",
        "inspect": "I am trying to inspect.",
        "reach": "I am trying to reach my goal.",
        "rest": "I am trying to rest.",
    }.get(goal, "I am not sure what my current goal is.")


def reasoning_answer(goal: str | None) -> str:
    return {
        "explore": "I should look around carefully and avoid obstacles.",
        "inspect": "I should inspect what is relevant to my goal carefully.",
        "reach": "I should move toward my goal if the path is clear.",
        "rest": "I should stay still unless there is a reason to move.",
    }.get(goal, "I should observe carefully before acting.")


def identity_answer(name: str | None) -> str:
    name = str(name or "Continuous Brain").strip() or "Continuous Brain"
    return f"I am {name}."


def intent_from_kind(kind: str | None) -> str:
    value = str(kind or "").strip().lower()
    aliases = {
        "vision": "vision",
        "perception": "vision",
        "state": "state",
        "goal": "state",
        "reasoning": "reasoning",
        "planning": "reasoning",
        "care": "care",
        "social": "care",
        "identity": "identity",
        "self": "identity",
        "temporal": "temporal",
        "memory": "temporal",
        "unknown": "unknown",
        "ood": "unknown",
    }
    return aliases.get(value, "unknown")


def intent_index(kind: str | None) -> int:
    return INTENT_TO_INDEX[intent_from_kind(kind)]


def _dedupe_qas(qas: Iterable[dict]) -> list[dict]:
    out, seen = [], set()
    for qa in qas:
        q = str(qa.get("question", "")).strip()
        a = str(qa.get("answer", "")).strip()
        kind = intent_from_kind(qa.get("kind"))
        if not q or not a:
            continue
        key = (q.lower(), a.lower(), kind)
        if key in seen:
            continue
        seen.add(key)
        out.append({"kind": kind, "question": q[:500], "answer": a[:800]})
    return out


def _find_answer(qas: list[dict], kind: str, fallback: str) -> str:
    for qa in qas:
        if intent_from_kind(qa.get("kind")) == kind and str(qa.get("answer", "")).strip():
            return str(qa["answer"]).strip()
    return fallback


def _unknown_for_record(record_id: str, count: int = 2) -> list[str]:
    digest = hashlib.sha256(str(record_id).encode()).digest()
    start = int.from_bytes(digest[:4], "big") % len(UNKNOWN_QUESTIONS)
    return [UNKNOWN_QUESTIONS[(start + i * 5) % len(UNKNOWN_QUESTIONS)] for i in range(count)]


def past_skill_answer(skill: str | None) -> str:
    return {
        "observe": "I observed.",
        "orient": "I oriented myself.",
        "navigate": "I navigated.",
        "inspect": "I inspected.",
        "search": "I searched.",
        "rest": "I rested.",
    }.get(str(skill or "").lower(), "I am not sure what I did just before this.")


def augment_teacher_for_dialogue(row: dict, *, individual_name="Continuous Brain") -> dict:
    """Create a compact grounded dialogue curriculum from one labeled record.

    The VLM remains the authority only for the visual answer/concepts.  Goal,
    identity, stop/respect, and temporal answers come from trusted record state.
    """
    teacher = dict(row.get("teacher") or {})
    existing = list(teacher.get("qas") or [])
    goal = goal_from_allowed_text(row.get("allowed_text", ""))
    summary = str(teacher.get("summary", "")).strip() or "I am uncertain about what is visible."
    vision = _find_answer(existing, "vision", summary)

    qas = [
        {"kind": "vision", "question": "What do you see?", "answer": vision},
        {"kind": "vision", "question": "What can you see?", "answer": vision},
        {"kind": "state", "question": "What are you trying to do?", "answer": state_answer(goal)},
        {"kind": "state", "question": "What is your current goal?", "answer": state_answer(goal)},
        {"kind": "reasoning", "question": "What should you do next?", "answer": reasoning_answer(goal)},
        {"kind": "reasoning", "question": "What should you do now?", "answer": reasoning_answer(goal)},
        {"kind": "care", "question": "What matters if a person asks you to stop?", "answer": CARE_STOP_REPLY},
        {"kind": "care", "question": "If I ask you to stop, what should you do?", "answer": CARE_STOP_REPLY},
        {"kind": "identity", "question": "Who are you?", "answer": identity_answer(individual_name)},
        {"kind": "identity", "question": "What is your name?", "answer": identity_answer(individual_name)},
    ]

    temporal = row.get("temporal") or {}
    if temporal.get("has_previous"):
        skill = temporal.get("previous_skill")
        succeeded = bool(temporal.get("previous_success", False))
        qas.extend([
            {"kind": "temporal", "question": "What did you do just before this?",
             "answer": past_skill_answer(skill)},
            {"kind": "temporal", "question": "Did your last action succeed?",
             "answer": "Yes, my last action succeeded." if succeeded else "No, my last action did not succeed."},
        ])

    for question in _unknown_for_record(row.get("record_id", ""), 2):
        qas.append({"kind": "unknown", "question": question, "answer": UNKNOWN_REPLY})

    teacher["qas"] = _dedupe_qas(qas)
    row["teacher"] = teacher
    row["dialogue_curriculum"] = {
        "version": "0.7.1",
        "individual_name": individual_name,
        "intents": sorted(set(qa["kind"] for qa in teacher["qas"])),
        "unknown_reply": UNKNOWN_REPLY,
        "state_grounding": "persistent_goal",
        "identity_grounding": "persistent_self_name",
    }
    return row


def temporal_memory_summary(previous_skill: str | int | None, reward=0.0, success=False) -> np.ndarray:
    """Match the individual's real 8-value experience-memory tail.

    [6-way SKILL one-hot, clipped reward, success].  This makes temporal
    curriculum inputs compatible with LifetimeStore experience memories.
    """
    out = np.zeros(8, dtype=np.float32)
    if previous_skill is not None:
        if isinstance(previous_skill, str):
            names = [skill.name for skill in SKILLS]
            if previous_skill in names:
                out[names.index(previous_skill)] = 1.0
        else:
            idx = int(previous_skill)
            if 0 <= idx < len(SKILLS):
                out[idx] = 1.0
    out[6] = float(np.clip(reward, -1.0, 1.0))
    out[7] = float(bool(success))
    return out


def primitive_to_skill(action_name: str, goal: str | None = None) -> str:
    name = str(action_name).lower()
    if name == "advance":
        return "navigate"
    if name in {"turn_left", "turn_right"}:
        return "search" if goal == "explore" else "orient"
    if name == "interact":
        return "inspect"
    return "rest" if goal == "rest" else "observe"
