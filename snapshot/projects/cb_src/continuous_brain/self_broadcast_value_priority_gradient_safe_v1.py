#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import torch
from torch import nn


class GradientSafeValuePriorityHead(nn.Module):
    """
    Nonlinear priority readout from the FROZEN shared self broadcast.

    No raw persistent-self fields are exposed.

    Exact-neutral and gradient-safe output:
        y = 2 * sigmoid(raw) - 1

    Therefore:
        raw == 0 -> y == 0 exactly
        dy/draw at raw=0 == 0.5

    Unlike clamp(), negative predictions retain gradient.
    """
    def __init__(
        self,
        broadcast_dim: int = 64,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.broadcast_dim = int(broadcast_dim)
        self.hidden_dim = int(hidden_dim)

        self.fc1 = nn.Linear(
            self.broadcast_dim,
            self.hidden_dim,
            bias=False,
        )
        self.fc2 = nn.Linear(
            self.hidden_dim,
            self.hidden_dim,
            bias=False,
        )
        self.fc3 = nn.Linear(
            self.hidden_dim,
            1,
            bias=False,
        )

    def forward(self, broadcast: torch.Tensor):
        if (
            broadcast.ndim != 2
            or broadcast.shape[-1] != self.broadcast_dim
        ):
            raise ValueError(
                f"broadcast must be [B,{self.broadcast_dim}], "
                f"got {tuple(broadcast.shape)}"
            )

        h = torch.nn.functional.silu(
            self.fc1(broadcast)
        )
        h = torch.nn.functional.silu(
            self.fc2(h)
        )
        raw = self.fc3(h).squeeze(-1)

        # Exactly zero for zero broadcast and nonzero gradient at zero.
        return 2.0 * torch.sigmoid(raw) - 1.0


def save_priority_head(path, head, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format":
                "continuous_brain.self_broadcast_value_priority_gradient_safe_v1",
            "broadcast_dim": int(head.broadcast_dim),
            "hidden_dim": int(head.hidden_dim),
            "state_dict": {
                k: v.detach().cpu()
                for k, v in head.state_dict().items()
            },
            "metadata": dict(metadata),
        },
        path,
    )


def load_priority_head(path, device="cpu"):
    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )
    head = GradientSafeValuePriorityHead(
        broadcast_dim=int(payload["broadcast_dim"]),
        hidden_dim=int(payload["hidden_dim"]),
    ).to(device)
    head.load_state_dict(payload["state_dict"])
    head.eval()
    return head, payload
