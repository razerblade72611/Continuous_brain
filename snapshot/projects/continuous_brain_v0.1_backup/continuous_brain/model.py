from __future__ import annotations

import torch
from torch import nn


class ContinuousBrainNet(nn.Module):
    """Perception + recurrent cognitive state + policy/Q head + predictive world model."""

    def __init__(self, obs_dim: int, action_dim: int, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.perception = nn.Sequential(
            nn.Linear(obs_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
        )
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
    ) -> torch.Tensor:
        emb = self.perception(obs)
        return self.core(torch.cat([emb, prev_action_onehot], dim=-1), hidden)

    def q_values(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.q_head(hidden)

    def predict(self, hidden: torch.Tensor, action_onehot: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([hidden, action_onehot], dim=-1)
        return self.world_head(x), self.reward_head(x).squeeze(-1)
