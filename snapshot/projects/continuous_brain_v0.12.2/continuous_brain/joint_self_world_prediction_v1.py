#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import torch
from torch import nn
from torch.nn import functional as F

from continuous_brain.persistent_self_autobiographical_v1 import (
    GOAL_NONE,
    GOAL_COMMITMENT_A,
    GOAL_COMMITMENT_B,
    GOAL_INVESTIGATE,
    ACTION_OBSERVE,
    ACTION_WORK_A,
    ACTION_WORK_B,
    ACTION_WAIT,
    EndogenousGoalArbiter,
    PersistentSelfState,
)

WORLD_DIM = 4
SELF_VECTOR_DIM = 10
ACTION_DIM = 4

# World vector:
# 0 progress_a
# 1 progress_b
# 2 information_quality
# 3 volatility


@dataclass
class JointTransition:
    current_world: torch.Tensor
    current_self: torch.Tensor
    action: int
    next_world: torch.Tensor
    next_self: torch.Tensor
    agency: float
    surprise: float


class JointSelfWorldModel(nn.Module):
    """
    Predicts future WORLD and future SELF jointly from:
      current world4 + current persistent-self vector10 + action4 one-hot.

    The self target is the actual v0.28 persistent self vector:
      unresolved A/B, confidence, uncertainty, prediction error, agency,
      active-goal one-hot4.

    This deliberately does not compress the persistent self through the failed
    v0.29 scalar-priority bottleneck.
    """
    def __init__(
        self,
        hidden_dim: int = 160,
        latent_dim: int = 96,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.latent_dim = int(latent_dim)

        self.encoder = nn.Sequential(
            nn.Linear(
                WORLD_DIM + SELF_VECTOR_DIM + ACTION_DIM,
                hidden_dim,
            ),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.Tanh(),
        )

        self.world_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, WORLD_DIM),
        )
        self.self_cont_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 6),
        )
        self.goal_head = nn.Linear(latent_dim, 4)
        self.agency_head = nn.Linear(latent_dim, 1)

        # Auxiliary reconstruction makes current self information explicitly
        # recoverable from the joint latent.
        self.current_self_reconstruction = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, SELF_VECTOR_DIM),
        )

    def forward(
        self,
        world: torch.Tensor,
        self_vector: torch.Tensor,
        action: torch.Tensor,
    ):
        action_oh = F.one_hot(
            action.long(),
            num_classes=ACTION_DIM,
        ).float()
        x = torch.cat(
            (world, self_vector, action_oh),
            dim=-1,
        )
        latent = self.encoder(x)

        world_next = torch.sigmoid(
            self.world_head(latent)
        )
        self_cont = torch.sigmoid(
            self.self_cont_head(latent)
        )
        goal_logits = self.goal_head(latent)
        agency_logit = self.agency_head(
            latent
        ).squeeze(-1)

        current_self_recon = torch.cat(
            (
                torch.sigmoid(
                    self.current_self_reconstruction(latent)[..., :6]
                ),
                self.current_self_reconstruction(latent)[..., 6:],
            ),
            dim=-1,
        )

        return {
            "latent": latent,
            "next_world": world_next,
            "next_self_continuous": self_cont,
            "next_goal_logits": goal_logits,
            "agency_logit": agency_logit,
            "current_self_reconstruction": current_self_recon,
        }


def vector_to_parts(v: torch.Tensor):
    if v.shape[-1] != SELF_VECTOR_DIM:
        raise ValueError(v.shape)
    return v[..., :6], v[..., 6:10]


def predicted_self_vector(out: dict):
    goal = F.softmax(
        out["next_goal_logits"],
        dim=-1,
    )
    return torch.cat(
        (
            out["next_self_continuous"],
            goal,
        ),
        dim=-1,
    )


def goal_from_vector(v: torch.Tensor):
    return int(
        v[..., 6:10].argmax(dim=-1).item()
    )


def build_self_state_from_vector(
    vector: torch.Tensor,
    *,
    identity_seed: int,
):
    v = vector.detach().cpu().float()
    if v.ndim != 1 or v.shape[0] != SELF_VECTOR_DIM:
        raise ValueError(v.shape)

    s = PersistentSelfState.create(
        identity_seed=int(identity_seed)
    )
    s.unresolved_a = float(v[0].item())
    s.unresolved_b = float(v[1].item())
    s.confidence = float(v[2].item())
    s.uncertainty = float(v[3].item())
    s.prediction_error_ema = float(v[4].item())
    s.agency_ema = float(v[5].item())
    s.active_goal = int(
        v[6:10].argmax().item()
    )
    return s


def utility_from_prediction(
    out: dict,
    *,
    active_goal: int,
    self_weight: float = 1.0,
):
    """
    Utility intentionally combines external and internal futures.

    World term:
      progress in the currently relevant commitment.

    Self term:
      lower unresolved commitment,
      lower uncertainty,
      lower prediction error,
      higher confidence.

    Information seeking receives value when active goal is INVESTIGATE.
    """
    world = out["next_world"]
    self_c = out["next_self_continuous"]
    goal_prob = F.softmax(
        out["next_goal_logits"],
        dim=-1,
    )

    if int(active_goal) == GOAL_COMMITMENT_A:
        world_value = world[..., 0]
        unresolved = self_c[..., 0]
    elif int(active_goal) == GOAL_COMMITMENT_B:
        world_value = world[..., 1]
        unresolved = self_c[..., 1]
    elif int(active_goal) == GOAL_INVESTIGATE:
        world_value = world[..., 2]
        unresolved = 0.5 * (
            self_c[..., 0]
            + self_c[..., 1]
        )
    else:
        world_value = 0.5 * (
            world[..., 0]
            + world[..., 1]
        )
        unresolved = 0.5 * (
            self_c[..., 0]
            + self_c[..., 1]
        )

    confidence = self_c[..., 2]
    uncertainty = self_c[..., 3]
    pred_error = self_c[..., 4]

    self_value = (
        0.40 * (1.0 - unresolved)
        + 0.20 * confidence
        + 0.20 * (1.0 - uncertainty)
        + 0.20 * (1.0 - pred_error)
    )

    if int(active_goal) == GOAL_INVESTIGATE:
        self_value = (
            self_value
            + 0.35 * goal_prob[..., GOAL_INVESTIGATE]
        )

    return world_value + float(self_weight) * self_value


@torch.no_grad()
def choose_action(
    model: JointSelfWorldModel,
    world: torch.Tensor,
    self_vector: torch.Tensor,
    *,
    self_weight: float = 1.0,
):
    if world.ndim == 1:
        world = world.unsqueeze(0)
    if self_vector.ndim == 1:
        self_vector = self_vector.unsqueeze(0)

    active_goal = int(
        self_vector[0, 6:10]
        .argmax()
        .item()
    )

    scores = []
    predictions = []
    for action in range(ACTION_DIM):
        a = torch.tensor(
            [action],
            dtype=torch.long,
            device=world.device,
        )
        out = model(
            world,
            self_vector,
            a,
        )
        score = float(
            utility_from_prediction(
                out,
                active_goal=active_goal,
                self_weight=self_weight,
            )[0].item()
        )
        scores.append(score)
        predictions.append(out)

    best = max(
        range(ACTION_DIM),
        key=lambda i: scores[i],
    )
    return {
        "action": int(best),
        "scores": scores,
        "predictions": predictions,
    }


def self_prediction_error(
    predicted_self: torch.Tensor,
    actual_self: torch.Tensor,
):
    """
    Scalar metacognitive error between predicted future self and actual self.

    Continuous fields and goal distribution both contribute.
    """
    if predicted_self.ndim > 1:
        predicted_self = predicted_self.squeeze(0)
    if actual_self.ndim > 1:
        actual_self = actual_self.squeeze(0)

    cont = (
        predicted_self[:6]
        - actual_self[:6]
    ).abs().mean()

    p = predicted_self[6:10]
    q = actual_self[6:10]
    goal = (p - q).abs().mean()

    return float(
        (0.75 * cont + 0.25 * goal).item()
    )


def save_joint_model(path, model, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format":
                "continuous_brain.joint_self_world_model_v1",
            "hidden_dim": int(model.hidden_dim),
            "latent_dim": int(model.latent_dim),
            "state_dict": {
                k: v.detach().cpu()
                for k, v in model.state_dict().items()
            },
            "metadata": dict(metadata),
        },
        path,
    )


def load_joint_model(path, device="cpu"):
    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )
    model = JointSelfWorldModel(
        hidden_dim=int(payload["hidden_dim"]),
        latent_dim=int(payload["latent_dim"]),
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload
