#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class ValuePriorityRepairHead(nn.Module):
    """
    Nonlinear readout from the already-frozen shared self broadcast.

    Bias-free by construction:
      broadcast == 0 -> hidden == 0 -> raw == 0 -> priority == 0 exactly.

    No raw persistent-self fields are visible here.  The repair can use only
    information already present in the admitted/frozen shared broadcast.
    """
    def __init__(
        self,
        broadcast_dim: int = 64,
        hidden_dim: int = 96,
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
        raw = self.fc2(h).squeeze(-1)

        # The target is in [0,1].  Clamping preserves an exact zero when the
        # shared broadcast is absent, unlike sigmoid(0)=0.5.
        return torch.clamp(raw, 0.0, 1.0)


def save_priority_head(path, head, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format":
                "continuous_brain.self_broadcast_value_priority_repair_v1",
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
    head = ValuePriorityRepairHead(
        broadcast_dim=int(payload["broadcast_dim"]),
        hidden_dim=int(payload["hidden_dim"]),
    ).to(device)
    head.load_state_dict(payload["state_dict"])
    head.eval()
    return head, payload
