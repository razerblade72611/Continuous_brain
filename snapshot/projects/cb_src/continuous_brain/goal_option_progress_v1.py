#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import torch
from torch import nn

from continuous_brain.additive_goal_options_v1 import (
    load_goal_options,
    sha256,
)

GOAL_OPTIONS_PATH = Path("state/additive_goal_options_v1/goal_options.pt")


class TemporalProgressHead(nn.Module):
    """
    Dense remaining-duration estimator for a frozen goal option.

    Input is the frozen AdditiveGoalOptionBank feature vector.
    Output is log1p(remaining lateral primitive steps).
    """
    def __init__(self, feature_dim: int):
        super().__init__()
        d = int(feature_dim)
        self.feature_dim = d
        self.net = nn.Sequential(
            nn.Linear(d, d),
            nn.SiLU(),
            nn.Linear(d, 1),
        )

    def forward(self, features):
        return self.net(features).squeeze(-1)


def save_progress_head(
    path,
    head,
    *,
    goal_options_sha256,
    goal_decoder_sha256,
    metadata,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "continuous_brain.goal_option_progress_v1",
            "goal_options_sha256": str(goal_options_sha256),
            "goal_decoder_sha256": str(goal_decoder_sha256),
            "feature_dim": int(head.feature_dim),
            "state_dict": {
                k: v.detach().cpu()
                for k, v in head.state_dict().items()
            },
            "metadata": dict(metadata),
        },
        path,
    )


def load_progress_head(
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

    head = TemporalProgressHead(
        int(payload["feature_dim"])
    ).to(device)
    head.load_state_dict(payload["state_dict"])
    head.eval()
    return head, payload
