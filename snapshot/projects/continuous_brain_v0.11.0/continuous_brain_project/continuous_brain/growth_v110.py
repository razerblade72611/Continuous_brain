from __future__ import annotations

"""v0.11.0 function-preserving neural growth primitives.

The pre-v0.11 child remains embedded exactly in the expanded network. Growth
branches are ordinary trainable residual MLPs whose output projection starts at
zero, therefore every branch is the identity map at migration time.  They are
child-owned capacity, not external parent weights.
"""

from dataclasses import dataclass
import torch
from torch import nn


class ZeroResidualBlock(nn.Module):
    """x -> x + f(x), with f(x)==0 exactly at initialization."""
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.in_proj = nn.Linear(int(dim), int(hidden))
        self.act = nn.GELU()
        self.out_proj = nn.Linear(int(hidden), int(dim))
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.out_proj(self.act(self.in_proj(x)))


class ZeroResidualStack(nn.Module):
    def __init__(self, dim: int, hidden: int, blocks: int):
        super().__init__()
        self.blocks = nn.ModuleList([
            ZeroResidualBlock(int(dim), int(hidden)) for _ in range(max(0, int(blocks)))
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class IdentityGrowth(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    @property
    def parameter_count(self) -> int:
        return 0


@dataclass(frozen=True)
class GrowthPlan:
    name: str = "balanced-v110"
    bus_hidden: int = 2048
    bus_blocks: int = 5
    audio_hidden: int = 1024
    audio_blocks: int = 2
    nucleus_hidden: int = 2048
    nucleus_blocks: int = 5
    language_hidden: int = 2048
    language_blocks: int = 8
    temporal_hidden: int = 1536
    temporal_blocks: int = 3
    world_hidden: int = 2048
    world_blocks: int = 5
    planning_hidden: int = 2048
    planning_blocks: int = 5
    option_hidden: int = 1536
    option_blocks: int = 3
    head_hidden: int = 1024
    head_blocks: int = 2
    fusion_hidden: int = 2048
    fusion_blocks: int = 5


def make_growth(enabled: bool, dim: int, hidden: int, blocks: int) -> nn.Module:
    if not enabled or int(blocks) <= 0:
        return IdentityGrowth()
    return ZeroResidualStack(int(dim), int(hidden), int(blocks))


def is_growth_parameter(name: str) -> bool:
    return ".growth." in f".{name}." or name.startswith("growth.") or ".growth_" in name or name.startswith("growth_")
