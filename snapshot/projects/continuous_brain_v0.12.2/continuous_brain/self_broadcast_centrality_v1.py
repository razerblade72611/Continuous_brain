#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class SelfBroadcastHub(nn.Module):
    """
    One shared persistent-self representation with multiple consumers.

    Input:
      identity anchor32 + persistent self vector10 = 42 dimensions.

    Shared broadcast latent:
      42 -> hidden -> broadcast_dim.

    Consumers:
      - actual workspace residual
      - autobiographical retrieval route
      - current-goal route
      - planning / information-gathering mode
      - planning-depth modulation
      - goal/value-priority modulation
      - identity reconstruction

    `recruitment=0` or `enabled=False` gives an EXACT zero broadcast at every
    consumer; this is the structural reactive-competence preservation path.
    """
    def __init__(
        self,
        workspace_dim: int,
        *,
        input_dim: int = 42,
        hidden_dim: int = 96,
        broadcast_dim: int = 64,
    ):
        super().__init__()
        self.workspace_dim = int(workspace_dim)
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.broadcast_dim = int(broadcast_dim)

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.broadcast_dim),
            nn.Tanh(),
        )

        # Bias-free consumers ensure zero shared broadcast -> exact neutral
        # outputs for every consumer.
        self.workspace = nn.Linear(
            self.broadcast_dim,
            self.workspace_dim,
            bias=False,
        )
        self.retrieval = nn.Linear(
            self.broadcast_dim,
            3,  # slot0 / slot1 / none
            bias=False,
        )
        self.goal = nn.Linear(
            self.broadcast_dim,
            4,  # NONE / A / B / INVESTIGATE
            bias=False,
        )
        self.planning_mode = nn.Linear(
            self.broadcast_dim,
            2,  # continue/execute vs investigate
            bias=False,
        )
        self.planning_depth = nn.Linear(
            self.broadcast_dim,
            1,
            bias=False,
        )
        self.value_priority = nn.Linear(
            self.broadcast_dim,
            1,
            bias=False,
        )
        self.identity_reconstruction = nn.Linear(
            self.broadcast_dim,
            32,
            bias=False,
        )

    def forward(
        self,
        self_input: torch.Tensor,
        *,
        recruitment: torch.Tensor | float | None = None,
        enabled: bool = True,
    ):
        if self_input.ndim != 2 or self_input.shape[-1] != self.input_dim:
            raise ValueError(
                f"self_input must be [B,{self.input_dim}], "
                f"got {tuple(self_input.shape)}"
            )

        latent = self.encoder(self_input)

        if recruitment is None:
            recruit = torch.ones(
                self_input.shape[0],
                1,
                dtype=self_input.dtype,
                device=self_input.device,
            )
        elif isinstance(recruitment, torch.Tensor):
            recruit = recruitment.to(
                device=self_input.device,
                dtype=self_input.dtype,
            )
            if recruit.ndim == 1:
                recruit = recruit.unsqueeze(-1)
            if recruit.shape != (self_input.shape[0], 1):
                raise ValueError(
                    "recruitment must be scalar, [B], or [B,1]"
                )
        else:
            recruit = torch.full(
                (self_input.shape[0], 1),
                float(recruitment),
                dtype=self_input.dtype,
                device=self_input.device,
            )

        if not enabled:
            recruit = torch.zeros_like(recruit)

        broadcast = latent * recruit

        return {
            "latent_pre_recruitment": latent,
            "broadcast": broadcast,
            "workspace_residual": self.workspace(broadcast),
            "retrieval_logits": self.retrieval(broadcast),
            "goal_logits": self.goal(broadcast),
            "planning_mode_logits":
                self.planning_mode(broadcast),
            "planning_depth":
                torch.sigmoid(
                    self.planning_depth(broadcast)
                ).squeeze(-1),
            "value_priority":
                torch.sigmoid(
                    self.value_priority(broadcast)
                ).squeeze(-1),
            "identity_reconstruction":
                F.normalize(
                    self.identity_reconstruction(broadcast),
                    dim=-1,
                    eps=1e-8,
                ),
        }


def self_input_from_state(state) -> torch.Tensor:
    identity = state.identity_embedding.detach().float()
    vector = state.vector().detach().float()
    if identity.ndim != 1 or identity.shape[0] != 32:
        raise ValueError("identity embedding must be [32]")
    if vector.ndim != 1 or vector.shape[0] != 10:
        raise ValueError("self state vector must be [10]")
    return torch.cat((identity, vector), dim=0)


def apply_workspace_broadcast(
    base,
    workspace: torch.Tensor,
    hub_output: dict,
):
    residual = hub_output["workspace_residual"]
    if workspace.shape != residual.shape:
        raise ValueError(
            f"workspace/residual mismatch: "
            f"{tuple(workspace.shape)} vs {tuple(residual.shape)}"
        )
    combined = workspace + residual
    logits = base.motor(combined)
    return combined, logits


def consumer_predictions(out: dict):
    return {
        "retrieval":
            out["retrieval_logits"].argmax(dim=-1),
        "goal":
            out["goal_logits"].argmax(dim=-1),
        "planning_mode":
            out["planning_mode_logits"].argmax(dim=-1),
    }


def save_hub(path, hub, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format":
                "continuous_brain.self_broadcast_hub_v1",
            "workspace_dim": int(hub.workspace_dim),
            "input_dim": int(hub.input_dim),
            "hidden_dim": int(hub.hidden_dim),
            "broadcast_dim": int(hub.broadcast_dim),
            "state_dict": {
                k: v.detach().cpu()
                for k, v in hub.state_dict().items()
            },
            "metadata": dict(metadata),
        },
        path,
    )


def load_hub(path, device="cpu"):
    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )
    hub = SelfBroadcastHub(
        workspace_dim=int(payload["workspace_dim"]),
        input_dim=int(payload["input_dim"]),
        hidden_dim=int(payload["hidden_dim"]),
        broadcast_dim=int(payload["broadcast_dim"]),
    ).to(device)
    hub.load_state_dict(payload["state_dict"])
    hub.eval()
    return hub, payload
