#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class FutureGoalRepairHead(nn.Module):
    """
    Calibration head over the FROZEN joint self/world model.

    Inputs are all outputs of the frozen model:
      joint latent96
      predicted continuous future self6
      original future-goal logits4

    No true future state is visible at inference.
    """
    def __init__(
        self,
        latent_dim: int = 96,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.input_dim = self.latent_dim + 6 + 4

        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 4),
        )

    def forward(self, frozen_out: dict):
        x = torch.cat(
            (
                frozen_out["latent"],
                frozen_out["next_self_continuous"],
                frozen_out["next_goal_logits"],
            ),
            dim=-1,
        )
        return self.net(x)


def fixed_world_utility(frozen_out: dict):
    """
    A genuinely self-blind world-only utility.

    It does not inspect current goal, commitments, or any predicted self field.
    """
    w = frozen_out["next_world"]

    # General external quality, independent of which commitment belongs to the
    # current self.
    return (
        0.425 * w[..., 0]
        + 0.425 * w[..., 1]
        + 0.175 * w[..., 2]
        - 0.025 * w[..., 3]
    )


def future_self_commitment_utility(
    frozen_out: dict,
    *,
    current_goal: int,
):
    """
    Utility of the predicted continuing self for the currently active
    commitment. This is the term removed by the corrected ablation.
    """
    s = frozen_out["next_self_continuous"]

    if int(current_goal) == 1:
        unresolved = s[..., 0]
    elif int(current_goal) == 2:
        unresolved = s[..., 1]
    else:
        unresolved = 0.5 * (
            s[..., 0] + s[..., 1]
        )

    confidence = s[..., 2]
    uncertainty = s[..., 3]
    pred_error = s[..., 4]

    return (
        0.72 * (1.0 - unresolved)
        + 0.10 * confidence
        + 0.09 * (1.0 - uncertainty)
        + 0.09 * (1.0 - pred_error)
    )


@torch.no_grad()
def corrected_choose_action(
    model,
    world: torch.Tensor,
    self_vector: torch.Tensor,
    *,
    include_future_self: bool,
    self_weight: float = 2.25,
):
    if world.ndim == 1:
        world = world.unsqueeze(0)
    if self_vector.ndim == 1:
        self_vector = self_vector.unsqueeze(0)

    current_goal = int(
        self_vector[0, 6:10]
        .argmax()
        .item()
    )

    scores = []
    for action in range(4):
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
        score = fixed_world_utility(out)

        if include_future_self:
            score = (
                score
                + float(self_weight)
                * future_self_commitment_utility(
                    out,
                    current_goal=current_goal,
                )
            )

        scores.append(float(score[0].item()))

    best = max(
        range(4),
        key=lambda i: scores[i],
    )
    return {
        "action": int(best),
        "scores": scores,
    }


def save_goal_head(path, head, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format":
                "continuous_brain.future_goal_repair_head_v1",
            "latent_dim": int(head.latent_dim),
            "hidden_dim": int(head.hidden_dim),
            "state_dict": {
                k: v.detach().cpu()
                for k, v in head.state_dict().items()
            },
            "metadata": dict(metadata),
        },
        path,
    )


def load_goal_head(path, device="cpu"):
    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )
    head = FutureGoalRepairHead(
        latent_dim=int(payload["latent_dim"]),
        hidden_dim=int(payload["hidden_dim"]),
    ).to(device)
    head.load_state_dict(payload["state_dict"])
    head.eval()
    return head, payload
