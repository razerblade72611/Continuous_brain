#!/usr/bin/env python3
from __future__ import annotations

from continuous_brain.dynamic_route_world_v1 import (
    ACTION_NAMES,
    LEFT_GOAL,
    RIGHT_GOAL,
    FAIL,
)
from continuous_brain.consequence_replanning_v1 import (
    predict_transition,
)


def plan_route_cost_consistent(
    model,
    state,
    spec,
    goal_side,
    *,
    max_depth=4,
    success_value=20.0,
    invalid_cutoff=.50,
    device="cpu",
):
    """
    Corrected v0.25 planner ranking.

    v0.25.0 sorted lexicographically by predicted success probability before
    utility.  Tiny calibration differences could therefore dominate route cost
    and select an unnecessary bypass, preventing the intended surprise event.

    Here predicted utility is primary:
        terminal_success_value * p(success) - cumulative_cost

    Success probability remains part of utility and a secondary tie-breaker.
    """
    beam = [{
        "state": dict(state),
        "actions": [],
        "predictions": [],
        "cost": 0.0,
        "score": 0.0,
        "terminal": False,
        "success_probability": 0.0,
    }]
    finished = []

    for _depth in range(int(max_depth)):
        expanded = []

        for node in beam:
            if node["terminal"]:
                finished.append(node)
                continue

            for action in range(len(ACTION_NAMES)):
                pred = predict_transition(
                    model,
                    node["state"],
                    spec,
                    goal_side,
                    action,
                    device=device,
                )
                if pred.valid_probability < float(invalid_cutoff):
                    continue

                total_cost = float(node["cost"]) + float(pred.cost)
                next_node = int(pred.next_state["node"])
                terminal = next_node in (LEFT_GOAL, RIGHT_GOAL, FAIL)

                score = (
                    float(success_value)
                    * float(pred.success_probability)
                    - total_cost
                )

                expanded.append({
                    "state": dict(pred.next_state),
                    "actions": list(node["actions"]) + [int(action)],
                    "predictions": list(node["predictions"]) + [{
                        "next_state": dict(pred.next_state),
                        "cost": float(pred.cost),
                        "success_probability":
                            float(pred.success_probability),
                        "valid_probability":
                            float(pred.valid_probability),
                        "next_blocked_probability":
                            float(pred.next_blocked_probability),
                    }],
                    "cost": total_cost,
                    "score": score,
                    "terminal": terminal,
                    "success_probability":
                        float(pred.success_probability),
                })

        if not expanded:
            break

        # Utility first.  Do not let calibration noise in p(success) outrank
        # a lower-cost route with the same predicted task outcome.
        expanded.sort(
            key=lambda n: (
                float(n["score"]),
                float(n["success_probability"]),
                float(n["terminal"]),
                -len(n["actions"]),
            ),
            reverse=True,
        )
        beam = expanded[:32]
        finished.extend(n for n in beam if n["terminal"])

    candidates = finished or beam
    if not candidates:
        raise RuntimeError("planner found no candidate")

    candidates.sort(
        key=lambda n: (
            float(n["score"]),
            float(n["success_probability"]),
            -len(n["actions"]),
        ),
        reverse=True,
    )
    best = candidates[0]

    if not best["actions"]:
        raise RuntimeError("planner returned empty action sequence")

    return {
        "actions": list(best["actions"]),
        "predictions": list(best["predictions"]),
        "score": float(best["score"]),
        "predicted_success_probability":
            float(best["success_probability"]),
        "predicted_cost": float(best["cost"]),
    }
