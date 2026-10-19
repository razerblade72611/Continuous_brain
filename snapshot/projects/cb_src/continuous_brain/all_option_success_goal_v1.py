#!/usr/bin/env python3
"""
Continuous Brain v0.20.2 — All-Option Success-Goal Planner v1

NO TRAINING. CPU reference backend.

Motivation
----------
v0.20.1 found:
- all-option goal swaps: ~23.7%
- selector-top2 goal swaps: ~10.5%
- selector-suppressed swaps: ~15.8%
- predicted success vs realized success: r ~ .92
- predicted collision vs realized collision: r ~ .24

Therefore:
1. remove selector-top2 as a hard candidate filter;
2. keep selector logits only as a tie-break/prior;
3. use the well-calibrated predicted-success head as the explicit goal term;
4. do NOT use predicted collision to steer the active goal until that head is
   recalibrated.

The neutral control reproduces reward + terminal option-Q ranking over the same
all-option candidate set.

The success goal adds only:
    success_weight * discounted predicted success probability

No learned tensor is changed.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable

import torch

from .infant import InfantBrain
from .planning_v103 import (
    _option_policy_action,
    _policy_tail_value,
)
from .goal_conditioned_option_planning_v1 import (
    GoalProfile,
    rollout_option_goal,
)

NEUTRAL_GOAL = GoalProfile(
    name="neutral",
    success_weight=0.0,
    collision_weight=0.0,
    goal_value_weight=0.0,
    q_tail_weight=1.0,
)

SUCCESS_GOAL = GoalProfile(
    name="success",
    success_weight=0.75,
    collision_weight=0.0,
    goal_value_weight=0.0,
    q_tail_weight=1.0,
)


def all_option_ids(model: InfantBrain):
    return list(range(int(model.cfg.option_count)))


@torch.no_grad()
def all_option_plan(
    model: InfantBrain,
    root: torch.Tensor,
    goal: GoalProfile,
    *,
    option_depth: int,
    beam_width: int,
    max_option_steps: int,
    discount: float,
    min_commit: int,
    terminate_threshold: float,
) -> dict:
    depth_limit = max(1, int(option_depth))
    width = max(1, int(beam_width))

    beam = [{
        "latent": root,
        "score": 0.0,
        "primitive_steps": 0,
        "options": [],
        "actions": [],
        "components": {
            "reward": 0.0,
            "success": 0.0,
            "collision": 0.0,
            "goal_value": 0.0,
        },
    }]

    option_ids = all_option_ids(model)

    for _depth in range(depth_limit):
        expanded = []

        for node in beam:
            for option in option_ids:
                macro = rollout_option_goal(
                    model,
                    node["latent"],
                    int(option),
                    goal,
                    max_option_steps=max_option_steps,
                    discount=discount,
                    min_commit=min_commit,
                    terminate_threshold=terminate_threshold,
                )

                offset = int(node["primitive_steps"])
                outer = float(discount) ** offset

                components = dict(node["components"])
                components["reward"] += (
                    outer * float(macro["reward_score"])
                )
                components["success"] += (
                    outer * float(macro["success_score"])
                )
                components["collision"] += (
                    outer * float(macro["collision_score"])
                )
                components["goal_value"] += (
                    outer * float(macro["learned_goal_score"])
                )

                base_score = components["reward"]
                goal_adjustment = (
                    float(goal.success_weight)
                    * components["success"]
                )

                expanded.append({
                    "latent": macro["latent"],
                    "score": float(base_score + goal_adjustment),
                    "base_score": float(base_score),
                    "goal_adjustment": float(goal_adjustment),
                    "primitive_steps":
                        offset + int(macro["primitive_steps"]),
                    "options":
                        list(node["options"]) + [int(option)],
                    "actions":
                        list(node["actions"]) + list(macro["actions"]),
                    "components": components,
                })

        if not expanded:
            break

        for node in expanded:
            tail = _policy_tail_value(
                model,
                node["latent"],
                int(model.cfg.option_count),
            )
            node["q_tail"] = float(tail)
            node["rank"] = (
                float(node["score"])
                + (float(discount) ** int(node["primitive_steps"]))
                * float(goal.q_tail_weight)
                * float(tail)
            )

        expanded.sort(
            key=lambda x: (
                x["rank"],
                x["components"]["success"],
                x["base_score"],
            ),
            reverse=True,
        )
        beam = expanded[:width]

    if not beam or not beam[0]["options"]:
        raise RuntimeError("all-option planner produced no plan")

    best = beam[0]

    return {
        "goal": asdict(goal),
        "chosen_option": int(best["options"][0]),
        "rank": float(best["rank"]),
        "imagined_options": list(best["options"]),
        "imagined_actions": list(best["actions"]),
        "imagined_primitive_steps": int(best["primitive_steps"]),
        "base_score": float(best["base_score"]),
        "goal_adjustment": float(best["goal_adjustment"]),
        "score_components": {
            **{
                k: float(v)
                for k, v in best["components"].items()
            },
            "q_tail": float(best.get("q_tail", 0.0)),
        },
    }
