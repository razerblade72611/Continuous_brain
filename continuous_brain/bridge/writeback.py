"""Write path (backbone -> core): the backbone's state becomes injected current.

A consultation does not replace the core's cognition; it creates an event in
the core's evolving state. The decoder maps a summary of the backbone's final
hidden states over its response to a bounded current on the core's input
neurons.

Training: spikes are not differentiable, so the decoder cannot be trained by
backprop through the core. It is a Gaussian policy instead: during waking,
exploration noise is added to the current and logged with the episode; during
sleep, REINFORCE with a baseline pushes the mean toward currents that preceded
better outcomes.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class WritebackDecoder(nn.Module):
    def __init__(self, hidden_size: int, n_input: int, hidden: int, max_current: float,
                 noise_std: float):
        super().__init__()
        self.max_current = max_current
        self.noise_std = noise_std
        self.net = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, hidden),
                                 nn.GELU(), nn.Linear(hidden, n_input))
        # Start near zero current: an untrained decoder should barely perturb the core.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    @staticmethod
    def summarize(hidden: torch.Tensor) -> torch.Tensor:
        """[L, H] response hidden states -> [H] float32 summary (stored with the episode)."""
        if hidden.shape[0] == 0:
            return torch.zeros(hidden.shape[-1])
        return hidden.float().mean(dim=0)

    def mean(self, summary: torch.Tensor) -> torch.Tensor:
        """[B, H] -> [B, n_input] mean current, bounded to +-max_current."""
        return torch.tanh(self.net(summary.float())) * self.max_current

    def sample(self, summary: torch.Tensor, generator: torch.Generator | None = None):
        mu = self.mean(summary)
        eps = torch.randn(mu.shape, generator=generator, device=mu.device)
        return mu + self.noise_std * eps, eps

    def log_prob(self, summary: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
        """Diagonal-Gaussian log-density of `current` under the policy. [B]."""
        mu = self.mean(summary)
        var = self.noise_std ** 2
        z = (current.float() - mu) ** 2 / var
        return -0.5 * (z + math.log(2 * math.pi * var)).sum(dim=-1)
