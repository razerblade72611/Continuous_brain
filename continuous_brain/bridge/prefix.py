"""Read path (core -> backbone): latent tokens become a soft-prompt prefix.

This is the BLIP-2 / Q-Former pattern recommended for v1 in the spec: the K
latent tokens are mapped into the backbone's embedding width and prepended to
the prompt, so the backbone's own unmodified self-attention reads them. No
new layers are spliced into the frozen model. (Interleaved Flamingo-style
cross-attention is a later upgrade.)

Each prefix token is rescaled to the backbone's typical embedding RMS so it
arrives in-distribution; a learned gain adjusts that during training.
"""

from __future__ import annotations

import torch
from torch import nn


class SoftPrefixBridge(nn.Module):
    def __init__(self, latent_dim: int, hidden_size: int, mlp_hidden: int, target_rms: float):
        super().__init__()
        self.mlp = nn.Sequential(nn.LayerNorm(latent_dim), nn.Linear(latent_dim, mlp_hidden),
                                 nn.GELU(), nn.Linear(mlp_hidden, hidden_size))
        self.register_buffer("target_rms", torch.tensor(float(target_rms)))
        self.log_gain = nn.Parameter(torch.zeros(()))

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        p = self.mlp(latents)
        rms = torch.sqrt((p * p).mean(dim=-1, keepdim=True) + 1e-6)
        return p / rms * self.target_rms * torch.exp(self.log_gain)
