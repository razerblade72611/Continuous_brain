#!/usr/bin/env python3
"""
Parameter-matched ungated recall control for Continuous Brain.

Purpose
-------
Match M2's recall parameter count exactly (49,280) without:
- recurrent-hidden/context input;
- contextual gating;
- bias terms that could learn a memory-independent correction.

Architecture
------------
memory[64]
  -> W1[64->256], bias=False
  -> tanh
  -> W2[256->128], bias=False, EXACTLY ZERO INITIALIZED
  -> static learned per-workspace scale[128], initialized to 1
  -> additive workspace residual

Parameter count:
  W1:    64*256   = 16,384
  W2:   256*128   = 32,768
  scale: 128      =    128
                     ------
                     49,280

At construction W2==0, therefore residual==0 exactly for arbitrary memory.
"""

from __future__ import annotations

import torch
from torch import nn

from continuous_brain.working_memory_recall_arms_v2 import (
    RecallBridgeWorkingMemoryInfant,
)


class CapacityMatchedUngatedRecallBridge(nn.Module):
    def __init__(self, memory_dim: int, bus_dim: int):
        super().__init__()
        if int(memory_dim) != 64:
            raise ValueError(
                "v1 control is pre-registered for memory_dim=64"
            )
        if int(bus_dim) != 128:
            raise ValueError(
                "v1 control is pre-registered for bus_dim=128"
            )

        self.memory_dim = int(memory_dim)
        self.bus_dim = int(bus_dim)
        self.decoder_dim = 256

        self.memory_features = nn.Linear(
            self.memory_dim,
            self.decoder_dim,
            bias=False,
        )
        self.output = nn.Linear(
            self.decoder_dim,
            self.bus_dim,
            bias=False,
        )
        self.output_scale = nn.Parameter(
            torch.ones(self.bus_dim)
        )

        # Deterministic standard initialization for the memory-only feature
        # transform; the exact no-op is guaranteed by zero output weights.
        nn.init.xavier_uniform_(self.memory_features.weight)
        nn.init.zeros_(self.output.weight)

    def forward(self, hidden, memory):
        del hidden  # explicitly no contextual/recurrent input
        features = torch.tanh(
            self.memory_features(memory)
        )
        decoded = self.output(features)
        residual = decoded * self.output_scale
        return residual, {
            "decoded": decoded,
            "recall_gate": None,
            "recall_multiplier": None,
            "memory_features": features,
        }


class CapacityMatchedRecallWorkingMemoryInfant(
    RecallBridgeWorkingMemoryInfant
):
    """Same writer/base wrapper as M1.5/M2, but ungated 49,280-param reader."""

    def __init__(self, base, cfg):
        # Build the same protected wrapper/writer, then replace only recall.
        super().__init__(base, cfg, arm="linear")
        self.arm = "capacity_ungated"
        self.recall = CapacityMatchedUngatedRecallBridge(
            int(cfg.memory_dim),
            int(cfg.bus_dim),
        )

    @classmethod
    def grow_from(cls, base, cfg):
        return cls(base, cfg)
