"""Neuron -> population pooling, then populations -> K learned latent tokens.

Stage 1 (fixed, structural): each projection neuron belongs to exactly one
population; features are mean-pooled with a sparse matrix, and a synchrony
channel is appended. This stage has no trainable parameters, so episodes can
store pooled features and be replayed during sleep without re-running the core.

Stage 2 (trained by backprop during sleep): a Perceiver / Q-Former style
cross-attention pool. K learned queries attend over the P populations. Before
projection, each population's feature vector is split into direction and
log-amplitude so overall activity level survives normalization.
"""

from __future__ import annotations

import torch
from torch import nn

from .features import population_synchrony
from .traces import TraceWindow


class PopulationAssignment:
    """A partition of N projection neurons into P non-empty populations."""

    def __init__(self, assignment: torch.Tensor, n_populations: int):
        assignment = assignment.long().cpu()
        if assignment.ndim != 1:
            raise ValueError("assignment must be 1-D: one population id per neuron")
        if int(assignment.min()) < 0 or int(assignment.max()) >= n_populations:
            raise ValueError("population id out of range")
        sizes = torch.bincount(assignment, minlength=n_populations)
        if bool((sizes == 0).any()):
            raise ValueError("every population needs at least one member neuron")
        self.assignment = assignment
        self.n_neurons = int(assignment.shape[0])
        self.n_populations = n_populations
        self.sizes = sizes
        rows = assignment
        cols = torch.arange(self.n_neurons)
        vals = 1.0 / sizes[rows].to(torch.float32)
        # Stored transposed ([P, N]) so torch.sparse.mm(A, dense) works directly.
        self._mean_t = torch.sparse_coo_tensor(torch.stack([rows, cols]), vals,
                                               (n_populations, self.n_neurons),
                                               check_invariants=True).coalesce()
        self._device_cache: dict[torch.device, torch.Tensor] = {}

    @classmethod
    def balanced(cls, n_neurons: int, n_populations: int, seed: int = 0) -> "PopulationAssignment":
        if n_populations > n_neurons:
            raise ValueError("more populations than neurons")
        g = torch.Generator().manual_seed(seed)
        ids = torch.arange(n_neurons) % n_populations
        return cls(ids[torch.randperm(n_neurons, generator=g)], n_populations)

    @classmethod
    def from_regions(cls, region_of_neuron: list[str]) -> "PopulationAssignment":
        """Use the core's own anatomical labels (e.g. InfantBrain streams) as populations."""
        names = sorted(set(region_of_neuron))
        index = {n: i for i, n in enumerate(names)}
        return cls(torch.tensor([index[r] for r in region_of_neuron]), len(names))

    def _mat(self, device: torch.device) -> torch.Tensor:
        if device not in self._device_cache:
            self._device_cache[device] = self._mean_t.to(device)
        return self._device_cache[device]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Mean over each population's members along the LAST dim: [..., N] -> [..., P]."""
        if x.shape[-1] != self.n_neurons:
            raise ValueError(f"expected last dim {self.n_neurons}, got {x.shape[-1]}")
        lead = x.shape[:-1]
        flat = x.reshape(-1, self.n_neurons).T                         # [N, R]
        out = torch.sparse.mm(self._mat(x.device), flat.to(torch.float32))  # [P, R]
        return out.T.reshape(*lead, self.n_populations)


class PopulationPooling:
    """[B, N, C] per-neuron features + window -> [B, P, C + 1] (last channel: synchrony)."""

    def __init__(self, assignment: PopulationAssignment):
        self.assignment = assignment

    def n_out_channels(self, n_in_channels: int) -> int:
        return n_in_channels + 1

    def __call__(self, feats: torch.Tensor, window: TraceWindow) -> torch.Tensor:
        pooled = self.assignment(feats.transpose(1, 2)).transpose(1, 2)   # [B, P, C]
        sync = population_synchrony(window.history_fast, self.assignment)  # [B, P]
        return torch.cat([pooled, sync[..., None]], dim=-1)


class LatentTokenPool(nn.Module):
    """[B, P, C] population features -> [B, K, D] latent tokens."""

    def __init__(self, in_channels: int, n_populations: int, n_tokens: int, dim: int,
                 n_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.in_channels = in_channels
        self.n_populations = n_populations
        self.in_proj = nn.Linear(in_channels + 1, dim)
        self.pop_embed = nn.Parameter(torch.randn(n_populations, dim) * 0.02)
        self.queries = nn.Parameter(torch.randn(n_tokens, dim) * 0.02)
        self.norm_kv = nn.LayerNorm(dim)
        self.norm_q = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(dropout),
                                nn.Linear(4 * dim, dim))

    @staticmethod
    def split_amplitude(x: torch.Tensor) -> torch.Tensor:
        amp = torch.sqrt((x * x).mean(dim=-1, keepdim=True) + 1e-6)
        return torch.cat([x / amp, torch.log1p(amp)], dim=-1)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        if pooled.shape[1:] != (self.n_populations, self.in_channels):
            raise ValueError(f"expected [B, {self.n_populations}, {self.in_channels}], "
                             f"got {tuple(pooled.shape)}")
        kv = self.norm_kv(self.in_proj(self.split_amplitude(pooled)) + self.pop_embed)
        q = self.queries.unsqueeze(0).expand(pooled.shape[0], -1, -1)
        z = q + self.attn(self.norm_q(q), kv, kv, need_weights=False)[0]
        return z + self.ff(self.norm_ff(z))
