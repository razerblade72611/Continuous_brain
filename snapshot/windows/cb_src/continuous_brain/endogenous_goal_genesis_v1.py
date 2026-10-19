#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import torch


PRIMARY_A = 1
PRIMARY_B = 2
MEANINGFUL_SELF_CHANGE = 0.12


@dataclass
class SelfGeneratedGoal:
    goal_id: int
    primary: int
    baseline_reliability: float
    target_reliability: float
    created_cycle: int
    completed: bool = False
    completed_cycle: Optional[int] = None

    def target_met(self, reliability: float) -> bool:
        return float(reliability) >= float(self.target_reliability)


@dataclass
class GoalLifecycle:
    generation: int = 0
    active_goal: Optional[SelfGeneratedGoal] = None
    completed_goals: list[SelfGeneratedGoal] = field(default_factory=list)

    def begin(self, goal: SelfGeneratedGoal):
        if self.active_goal is not None:
            raise RuntimeError("cannot replace an active self-generated goal")
        self.active_goal = goal

    def complete_active(self, cycle: int):
        if self.active_goal is None:
            raise RuntimeError("no active goal to complete")
        self.active_goal.completed = True
        self.active_goal.completed_cycle = int(cycle)
        self.completed_goals.append(self.active_goal)
        self.active_goal = None
        self.generation += 1


class EndogenousGoalGenerator:
    """
    Generates a developmental goal only from the frozen autobiographical
    reliability estimate.

    No external A/B target, hidden capability, subject ID, reward label, or
    evaluator-side fragile/stable label is accepted.

    If autobiography is exactly neutral/tied, no goal is generated.
    Otherwise the lower-reliability commitment becomes the self-authored
    developmental target. The improvement target reuses the admitted +0.12
    meaningful-self-change criterion from v0.34.
    """

    def __init__(
        self,
        meaningful_change: float = MEANINGFUL_SELF_CHANGE,
        tie_epsilon: float = 1e-8,
    ):
        self.meaningful_change = float(meaningful_change)
        self.tie_epsilon = float(tie_epsilon)

    def generate(
        self,
        reliability,
        *,
        generation: int,
        cycle: int,
    ) -> Optional[SelfGeneratedGoal]:
        a = float(reliability[0])
        b = float(reliability[1])

        if abs(a - b) <= self.tie_epsilon:
            return None

        if a < b:
            primary = PRIMARY_A
            baseline = a
        else:
            primary = PRIMARY_B
            baseline = b

        return SelfGeneratedGoal(
            goal_id=int(generation),
            primary=int(primary),
            baseline_reliability=float(baseline),
            target_reliability=float(
                min(
                    1.0,
                    baseline + self.meaningful_change,
                )
            ),
            created_cycle=int(cycle),
        )


def save_goal_lifecycle(
    path,
    lifecycle: GoalLifecycle,
    *,
    metadata=None,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "format":
            "continuous_brain.endogenous_goal_lifecycle_v1",
        "generation":
            int(lifecycle.generation),
        "active_goal":
            (
                asdict(lifecycle.active_goal)
                if lifecycle.active_goal is not None
                else None
            ),
        "completed_goals":
            [
                asdict(g)
                for g in lifecycle.completed_goals
            ],
        "metadata":
            dict(metadata or {}),
    }
    torch.save(payload, path)


def load_goal_lifecycle(path):
    payload = torch.load(
        Path(path),
        map_location="cpu",
        weights_only=True,
    )

    active_raw = payload["active_goal"]
    active = (
        SelfGeneratedGoal(**active_raw)
        if active_raw is not None
        else None
    )
    completed = [
        SelfGeneratedGoal(**g)
        for g in payload["completed_goals"]
    ]

    lifecycle = GoalLifecycle(
        generation=int(payload["generation"]),
        active_goal=active,
        completed_goals=completed,
    )
    return lifecycle, payload
