#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from continuous_brain.dynamic_route_world_v1 import (
    ACTION_NAMES,
    NODE_NAMES,
    FAIL,
    LEFT_GOAL,
    RIGHT_GOAL,
)


def state_vector(state: dict, spec) -> torch.Tensor:
    """
    Structured planning state for this mechanism assay.

    This is deliberately not claimed to be the historical PlanningCortex
    latent.  The sidecar isolates consequence prediction/replanning semantics.
    """
    node = F.one_hot(
        torch.tensor(int(state["node"]), dtype=torch.long),
        num_classes=len(NODE_NAMES),
    ).float()
    flags = torch.tensor(
        [
            float(bool(state["blocked"])),
            float(bool(state["known_target_blocked"])),
        ],
        dtype=torch.float32,
    )
    costs = torch.tensor(
        [
            float(spec.approach_left_cost),
            float(spec.approach_right_cost),
            float(spec.direct_cost),
            float(spec.switch_cost),
            float(spec.cross_cost),
        ],
        dtype=torch.float32,
    )
    return torch.cat((node, flags, costs), dim=0)


def goal_vector(goal_side: int) -> torch.Tensor:
    return F.one_hot(
        torch.tensor(int(goal_side), dtype=torch.long),
        num_classes=2,
    ).float()


def action_vector(action: int) -> torch.Tensor:
    return F.one_hot(
        torch.tensor(int(action), dtype=torch.long),
        num_classes=len(ACTION_NAMES),
    ).float()


class RouteConsequenceModel(nn.Module):
    """
    Predict one macro transition.

    Outputs:
      next node distribution
      next blocked probability
      immediate macro cost
      terminal goal-success probability
      action-validity probability
    """
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.node_dim = len(NODE_NAMES)
        self.action_dim = len(ACTION_NAMES)
        self.state_dim = self.node_dim + 2 + 5
        self.goal_dim = 2
        self.hidden_dim = int(hidden_dim)

        inp = self.state_dim + self.goal_dim + self.action_dim
        self.trunk = nn.Sequential(
            nn.Linear(inp, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.next_node = nn.Linear(hidden_dim, self.node_dim)
        self.next_blocked = nn.Linear(hidden_dim, 1)
        self.cost = nn.Linear(hidden_dim, 1)
        self.success = nn.Linear(hidden_dim, 1)
        self.valid = nn.Linear(hidden_dim, 1)

    def forward(self, state_v, goal_v, action_v):
        x = torch.cat((state_v, goal_v, action_v), dim=-1)
        h = self.trunk(x)
        return {
            "next_node_logits": self.next_node(h),
            "next_blocked_logit": self.next_blocked(h).squeeze(-1),
            "cost": F.softplus(self.cost(h).squeeze(-1)),
            "success_logit": self.success(h).squeeze(-1),
            "valid_logit": self.valid(h).squeeze(-1),
        }


@dataclass
class PredictedTransition:
    next_state: dict
    cost: float
    success_probability: float
    valid_probability: float
    next_blocked_probability: float


@torch.no_grad()
def predict_transition(model, state, spec, goal_side, action, device="cpu"):
    sv = state_vector(state, spec).to(device).unsqueeze(0)
    gv = goal_vector(goal_side).to(device).unsqueeze(0)
    av = action_vector(action).to(device).unsqueeze(0)

    out = model(sv, gv, av)
    next_node = int(torch.argmax(out["next_node_logits"], dim=-1)[0].item())
    blocked_p = float(torch.sigmoid(out["next_blocked_logit"])[0].item())

    return PredictedTransition(
        next_state={
            "node": next_node,
            "blocked": bool(blocked_p >= .5),
            "known_target_blocked": bool(state["known_target_blocked"]),
        },
        cost=float(out["cost"][0].item()),
        success_probability=float(
            torch.sigmoid(out["success_logit"])[0].item()
        ),
        valid_probability=float(
            torch.sigmoid(out["valid_logit"])[0].item()
        ),
        next_blocked_probability=blocked_p,
    )


@torch.no_grad()
def plan_route(
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
    Explicit short-horizon search over learned macro consequences.
    """
    beam = [{
        "state": dict(state),
        "actions": [],
        "predictions": [],
        "cost": 0.0,
        "score": -1e9,
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

                child = {
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
                }
                expanded.append(child)

        if not expanded:
            break

        # Favor terminal success, then model score, then shorter plan.
        expanded.sort(
            key=lambda n: (
                float(n["terminal"]),
                float(n["success_probability"]),
                float(n["score"]),
                -len(n["actions"]),
            ),
            reverse=True,
        )
        beam = expanded[:32]

        finished.extend([n for n in beam if n["terminal"]])

    candidates = finished or beam
    if not candidates:
        raise RuntimeError("planner found no candidate")

    candidates.sort(
        key=lambda n: (
            float(n["success_probability"]),
            float(n["score"]),
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


def state_mismatch(predicted: dict, actual: dict) -> bool:
    return (
        int(predicted["node"]) != int(actual["node"])
        or bool(predicted["blocked"]) != bool(actual["blocked"])
        or bool(predicted["known_target_blocked"])
        != bool(actual["known_target_blocked"])
    )


def save_consequence_model(path, model, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format":
                "continuous_brain.route_consequence_model_v1",
            "hidden_dim": int(model.hidden_dim),
            "state_dict": {
                k: v.detach().cpu()
                for k, v in model.state_dict().items()
            },
            "metadata": dict(metadata),
        },
        path,
    )


def load_consequence_model(path, device="cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = RouteConsequenceModel(
        hidden_dim=int(payload["hidden_dim"])
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload
