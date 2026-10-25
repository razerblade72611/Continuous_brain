"""Arm S: sparse product-key slot memory with Meta's sparse-memory-finetuning update rule.

Lookup (product keys, Lample et al. 2019; "Memory Layers at Scale", Berges et al. 2024):
    q = LayerNorm(h) @ Wq, split into halves q1, q2
    s1 = q1 @ K1.T, s2 = q2 @ K2.T          (n_keys sub-keys each -> n_keys**2 slots)
    top-k of s1 and of s2 -> k*k candidate pairs -> top-k pairs by s1+s2 -> softmax weights w
    out = sum_i w_i * V[slot_i]             (added to the residual stream)
Wq, K1, K2 are drawn from the birth seed and stay fixed; only the value table V learns. V is born at ZERO, so an
unlived memory contributes nothing, as in v2.

Update rule (Lin et al. 2025, arXiv 2510.15103): for each new piece of text, rank the slots it touches by
TF-IDF: tf = how often the text's tokens selected the slot, idf = log((N_bg + 1) / (bg_count(slot) + 1)),
where bg_count comes from a pass over never-scored background text at birth. Only the top-t slots by TF-IDF get
a gradient step; every other row of V is untouched. Slots that ordinary language uses a lot are protected.
"""
from __future__ import annotations

import hashlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SlotMemory(nn.Module):
    def __init__(self, d: int, n_keys: int = 48, k: int = 16, d_key: int = 256, birth_seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(birth_seed)
        self.d, self.n_keys, self.k, self.d_key = d, n_keys, k, d_key
        self.register_buffer("Wq", torch.randn(d, d_key, generator=g) / d ** 0.5)
        self.register_buffer("K1", F.normalize(torch.randn(n_keys, d_key // 2, generator=g), dim=-1))
        self.register_buffer("K2", F.normalize(torch.randn(n_keys, d_key // 2, generator=g), dim=-1))
        self.V = nn.Parameter(torch.zeros(n_keys * n_keys, d))
        self.register_buffer("bg_count", torch.zeros(n_keys * n_keys))
        self.register_buffer("bg_total", torch.zeros(()))
        self.momentum_state = torch.zeros_like(self.V)
        self.last_slots = None          # (tokens, k) slot ids of the most recent forward, for TF-IDF

    @property
    def n_slots(self):
        return self.n_keys * self.n_keys

    def lookup(self, x: torch.Tensor):
        """x: (..., d) -> slot ids (..., k) and softmax weights (..., k). Pure function of the fixed keys."""
        q = x @ self.Wq
        q1, q2 = q.split(self.d_key // 2, dim=-1)
        s1, i1 = (q1 @ self.K1.T).topk(self.k, dim=-1)
        s2, i2 = (q2 @ self.K2.T).topk(self.k, dim=-1)
        pair = (s1.unsqueeze(-1) + s2.unsqueeze(-2)).flatten(-2)              # (..., k*k)
        ids = (i1.unsqueeze(-1) * self.n_keys + i2.unsqueeze(-2)).flatten(-2)
        top, j = pair.topk(self.k, dim=-1)
        return ids.gather(-1, j), torch.softmax(top, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ids, w = self.lookup(x)
        self.last_slots = ids.detach().reshape(-1, self.k)
        return (w.unsqueeze(-1) * self.V[ids]).sum(-2)

    @torch.no_grad()
    def observe_background(self, x: torch.Tensor) -> None:
        """Accumulate background slot usage (birth-time pass over never-scored text). x: (tokens, d)."""
        ids, _ = self.lookup(x)
        self.bg_count += torch.bincount(ids.reshape(-1), minlength=self.n_slots).float()
        self.bg_total += ids.shape[0]

    def tfidf_slots(self, top_t: int) -> torch.Tensor:
        """Top-t slots for the text of the most recent forward pass, ranked by TF-IDF against background."""
        tf = torch.bincount(self.last_slots.reshape(-1), minlength=self.n_slots).float()
        idf = torch.log((self.bg_total + 1) / (self.bg_count + 1))
        score = tf * idf
        score[tf == 0] = -math.inf
        t = min(top_t, int((tf > 0).sum()))
        return score.topk(t).indices

    @torch.no_grad()
    def sparse_step(self, top_t: int, lr: float, momentum: float = 0.0) -> torch.Tensor:
        """Apply the gradient in V.grad to the top-t TF-IDF rows only; every other row stays bit-identical."""
        rows = self.tfidf_slots(top_t)
        g = self.V.grad
        self.momentum_state[rows] = momentum * self.momentum_state[rows] - lr * g[rows]
        self.V[rows] += self.momentum_state[rows]
        self.V.grad = None
        return rows

    def trainable_params(self) -> int:
        return self.V.numel()

    def weight_hash(self) -> str:
        h = hashlib.sha256()
        for n, p in sorted(list(self.named_parameters()) + list(self.named_buffers())):
            h.update(n.encode())
            h.update(p.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
