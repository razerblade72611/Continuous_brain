"""The set of stateful components, and what counts as the system's state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .bridge.backbone import Backbone
from .bridge.prefix import SoftPrefixBridge
from .bridge.writeback import WritebackDecoder
from .core.interface import SpikingCore
from .transducer.pooling import LatentTokenPool


@dataclass
class BrainComponents:
    core: SpikingCore
    backbone: Backbone              # frozen: never part of saved state
    latent_pool: LatentTokenPool    # trained during sleep (backprop)
    prefix: SoftPrefixBridge        # trained during sleep (backprop)
    writeback: WritebackDecoder     # trained during sleep (policy gradient)

    def bridge_parameters(self) -> list[torch.nn.Parameter]:
        return list(self.latent_pool.parameters()) + list(self.prefix.parameters())

    def prefix_from_features(self, features: torch.Tensor) -> torch.Tensor:
        """[B, P, C] pooled population features -> [B, K, H] soft prefix."""
        dev = next(self.latent_pool.parameters()).device
        return self.prefix(self.latent_pool(features.to(dev)))

    def full_state(self) -> dict[str, Any]:
        return {"core": self.core.state_dict(),
                "latent_pool": self.latent_pool.state_dict(),
                "prefix": self.prefix.state_dict(),
                "writeback": self.writeback.state_dict()}

    def load_full_state(self, state: dict[str, Any]) -> None:
        self.core.load_state_dict(state["core"])
        self.latent_pool.load_state_dict(state["latent_pool"])
        self.prefix.load_state_dict(state["prefix"])
        self.writeback.load_state_dict(state["writeback"])
