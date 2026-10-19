#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import torch
from torch import nn

GOAL_OPTIONS_PATH = Path("state/additive_goal_options_v1/goal_options.pt")
GOAL_DECODER_PATH = Path("state/memory_guided_goal_option_v1/goal_decoder.pt")


class RecurrentGoalOptionState(nn.Module):
    """Recurrent temporal state for a frozen goal-conditioned option."""
    def __init__(self, feature_dim: int, hidden_dim: int = 96):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.input_norm = nn.LayerNorm(self.feature_dim)
        self.gru = nn.GRU(
            input_size=self.feature_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.remaining_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.termination_head = nn.Linear(self.hidden_dim, 1)

    def initial_state(self, batch_size: int, device):
        return torch.zeros(
            1, int(batch_size), self.hidden_dim, device=device
        )

    def forward_sequence(self, features, hidden=None):
        x = self.input_norm(features)
        out, hidden_next = self.gru(x, hidden)
        return {
            "remaining_log": self.remaining_head(out).squeeze(-1),
            "termination_logit": self.termination_head(out).squeeze(-1),
            "hidden": hidden_next,
        }

    def step(self, feature, hidden=None):
        if feature.ndim != 2:
            raise ValueError(
                f"expected [B,D], got {tuple(feature.shape)}"
            )
        out = self.forward_sequence(feature.unsqueeze(1), hidden)
        return {
            "remaining_log": out["remaining_log"][:, 0],
            "termination_logit": out["termination_logit"][:, 0],
            "hidden": out["hidden"],
        }


def save_controller(
    path,
    controller,
    *,
    goal_options_sha256,
    goal_decoder_sha256,
    metadata,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format":
                "continuous_brain.recurrent_goal_option_state_v1",
            "goal_options_sha256": str(goal_options_sha256),
            "goal_decoder_sha256": str(goal_decoder_sha256),
            "feature_dim": int(controller.feature_dim),
            "hidden_dim": int(controller.hidden_dim),
            "state_dict": {
                k: v.detach().cpu()
                for k, v in controller.state_dict().items()
            },
            "metadata": dict(metadata),
        },
        path,
    )


def load_controller(
    path,
    *,
    expected_goal_options_sha256,
    expected_goal_decoder_sha256,
    device="cpu",
):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["goal_options_sha256"] != expected_goal_options_sha256:
        raise RuntimeError("goal option provenance mismatch")
    if payload["goal_decoder_sha256"] != expected_goal_decoder_sha256:
        raise RuntimeError("goal decoder provenance mismatch")
    controller = RecurrentGoalOptionState(
        int(payload["feature_dim"]),
        int(payload["hidden_dim"]),
    ).to(device)
    controller.load_state_dict(payload["state_dict"])
    controller.eval()
    return controller, payload
