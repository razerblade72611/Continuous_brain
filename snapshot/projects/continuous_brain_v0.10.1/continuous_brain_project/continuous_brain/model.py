from __future__ import annotations

import torch
from torch import nn


class ContinuousBrainNet(nn.Module):
    """State perception + optional foundation-vision adapter + recurrent cognition."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        embed_dim: int,
        hidden_dim: int,
        visual_dim: int = 0,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.visual_dim = int(visual_dim)

        # Keep v0.1/v0.2 names and dimensions unchanged for checkpoint migration.
        self.perception = nn.Sequential(
            nn.Linear(obs_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
        )
        if self.visual_dim > 0:
            self.visual_adapter = nn.Sequential(
                nn.Linear(self.visual_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim),
            )
            # Start as a small residual so a migrated v0.2 policy is not suddenly overwhelmed.
            self.visual_gate_logit = nn.Parameter(torch.tensor(-3.0))
        else:
            self.visual_adapter = None
            self.register_parameter("visual_gate_logit", None)

        self.core = nn.GRUCell(embed_dim + action_dim, hidden_dim)
        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, action_dim)
        )
        self.world_head = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, obs_dim),
        )
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def encode_step(
        self,
        obs: torch.Tensor,
        prev_action_onehot: torch.Tensor,
        hidden: torch.Tensor,
        visual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        emb = self.perception(obs)
        if self.visual_adapter is not None and visual is not None:
            v = self.visual_adapter(visual)
            gate = torch.sigmoid(self.visual_gate_logit)
            emb = emb + gate * v
        return self.core(torch.cat([emb, prev_action_onehot], dim=-1), hidden)

    def q_values(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.q_head(hidden)

    def predict(self, hidden: torch.Tensor, action_onehot: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([hidden, action_onehot], dim=-1)
        return self.world_head(x), self.reward_head(x).squeeze(-1)

    def visual_gate(self) -> float:
        if self.visual_gate_logit is None:
            return 0.0
        return float(torch.sigmoid(self.visual_gate_logit.detach()).cpu().item())
