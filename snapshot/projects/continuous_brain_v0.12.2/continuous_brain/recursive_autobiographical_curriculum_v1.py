#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch

from continuous_brain.endogenous_goal_genesis_v1 import (
    SelfGeneratedGoal,
)


MIN_SAME_PRIMARY_TARGET_ADVANCE = 0.05


def goal_semantics(goal: Optional[SelfGeneratedGoal]):
    if goal is None:
        return None
    return {
        "primary": int(goal.primary),
        "baseline_reliability": float(goal.baseline_reliability),
        "target_reliability": float(goal.target_reliability),
    }


def same_goal_semantics(
    a: Optional[SelfGeneratedGoal],
    b: Optional[SelfGeneratedGoal],
    *,
    atol: float = 1e-8,
) -> bool:
    if a is None or b is None:
        return a is b

    return (
        int(a.primary) == int(b.primary)
        and abs(
            float(a.baseline_reliability)
            - float(b.baseline_reliability)
        ) <= atol
        and abs(
            float(a.target_reliability)
            - float(b.target_reliability)
        ) <= atol
    )


def goal_is_advanced(
    previous: SelfGeneratedGoal,
    current: Optional[SelfGeneratedGoal],
    *,
    min_same_primary_target_advance: float = (
        MIN_SAME_PRIMARY_TARGET_ADVANCE
    ),
) -> bool:
    if current is None:
        return False

    if int(current.primary) != int(previous.primary):
        return True

    return (
        float(current.target_reliability)
        >= float(previous.target_reliability)
        + float(min_same_primary_target_advance)
    )


@dataclass
class CurriculumSnapshot:
    first_goal: SelfGeneratedGoal
    second_goal: Optional[SelfGeneratedGoal]
    frozen_history_second_goal: Optional[SelfGeneratedGoal]

    def to_payload(self):
        return {
            "first_goal": asdict(self.first_goal),
            "second_goal": (
                asdict(self.second_goal)
                if self.second_goal is not None
                else None
            ),
            "frozen_history_second_goal": (
                asdict(self.frozen_history_second_goal)
                if self.frozen_history_second_goal is not None
                else None
            ),
        }


def save_curriculum_snapshot(path, snapshot: CurriculumSnapshot):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format":
                "continuous_brain.recursive_autobiographical_curriculum_v1",
            "snapshot":
                snapshot.to_payload(),
        },
        path,
    )


def _goal_from_payload(raw):
    if raw is None:
        return None
    return SelfGeneratedGoal(**raw)


def load_curriculum_snapshot(path):
    payload = torch.load(
        Path(path),
        map_location="cpu",
        weights_only=True,
    )
    raw = payload["snapshot"]
    snapshot = CurriculumSnapshot(
        first_goal=SelfGeneratedGoal(
            **raw["first_goal"]
        ),
        second_goal=_goal_from_payload(
            raw["second_goal"]
        ),
        frozen_history_second_goal=_goal_from_payload(
            raw["frozen_history_second_goal"]
        ),
    )
    return snapshot, payload
