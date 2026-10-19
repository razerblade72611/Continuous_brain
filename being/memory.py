"""Titans-style neural memory: a small MLP whose weights are the being's memory.

Update rule (Behrouz et al., "Titans: Learning to Memorize at Test Time", 2024):
    S <- momentum * S - lr * grad(surprise loss)
    W <- (1 - forget) * W + S
The output layer is born at zero, so an unlived memory contributes nothing.
"""
from __future__ import annotations

import hashlib

import torch
import torch.nn as nn
import torch.nn.functional as F


class TitansMemory(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, d_out: int, birth_seed: int, train_input: bool = True):
        super().__init__()
        g = torch.Generator().manual_seed(birth_seed)
        self.W1 = nn.Parameter(torch.randn(d_hidden, d_in, generator=g) / d_in ** 0.5, requires_grad=train_input)
        self.b1 = nn.Parameter(torch.zeros(d_hidden), requires_grad=train_input)
        self.W2 = nn.Parameter(torch.zeros(d_out, d_hidden))
        self.momentum_state = {n: torch.zeros_like(p) for n, p in self.named_parameters() if p.requires_grad}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x @ self.W1.T + self.b1) @ self.W2.T

    def trainable(self):
        return [(n, p) for n, p in self.named_parameters() if p.requires_grad]

    @torch.no_grad()
    def titans_step(self, lr: float, momentum: float, forget: float) -> None:
        for n, p in self.trainable():
            if p.grad is None:
                continue
            s = self.momentum_state[n]
            s.mul_(momentum).sub_(lr * p.grad)
            p.mul_(1.0 - forget).add_(s)
            p.grad = None

    def snapshot(self) -> dict:
        return {
            "params": {n: p.detach().clone() for n, p in self.named_parameters()},
            "momentum": {n: s.clone() for n, s in self.momentum_state.items()},
        }

    @torch.no_grad()
    def restore(self, snap: dict) -> None:
        for n, p in self.named_parameters():
            p.copy_(snap["params"][n])
        for n in self.momentum_state:
            self.momentum_state[n].copy_(snap["momentum"][n])

    def weight_hash(self) -> str:
        h = hashlib.sha256()
        for n, p in sorted(self.named_parameters()):
            h.update(n.encode())
            h.update(p.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()


def module_hash(module: nn.Module) -> str:
    h = hashlib.sha256()
    for n, p in sorted(module.named_parameters()):
        h.update(n.encode())
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
