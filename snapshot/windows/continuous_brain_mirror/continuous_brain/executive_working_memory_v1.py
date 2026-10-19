#!/usr/bin/env python3
"""
Task-conditioned executive composition on top of the DEV-admitted reusable
working-memory system.

The admitted memory system is treated as a protected substrate:
- historical 14k base frozen;
- admitted 64-D writer frozen;
- admitted capacity-ungated recall reader frozen.

The new executive receives only:
    current 64-D stored memory
    late instruction ID

It does NOT receive the cue image, context vector, recurrent hidden state, or a
direct motor shortcut.

Architecture
------------
memory64 --bias-free--> memory_features128 --tanh--+
                                                    * --> joint128
instruction embedding128 ------------------tanh----+
                                                    |
                                              zero-init
                                              Linear128->128
                                                    |
                                            workspace residual

The residual is added to the admitted shared workspace before the frozen motor.
The world model continues to consume the admitted pre-executive workspace,
keeping goal semantics separate from physical-dynamics prediction.

Instruction NONE is hard-masked to an exact zero residual even after training.
This preserves all pre-instruction behavior exactly.
"""
from __future__ import annotations

import torch
from torch import nn

from .late_instruction_detour_world_v1 import INSTRUCTION_NONE


class LateBoundMemoryExecutive(nn.Module):
    def __init__(
        self,
        *,
        memory_dim: int = 64,
        bus_dim: int = 128,
        instruction_count: int = 3,
    ):
        super().__init__()
        self.memory_dim = int(memory_dim)
        self.bus_dim = int(bus_dim)
        self.instruction_count = int(instruction_count)

        self.memory_features = nn.Linear(
            self.memory_dim,
            self.bus_dim,
            bias=False,
        )
        self.instruction = nn.Embedding(
            self.instruction_count,
            self.bus_dim,
        )
        self.output = nn.Linear(
            self.bus_dim,
            self.bus_dim,
            bias=False,
        )

        nn.init.xavier_uniform_(self.memory_features.weight)
        nn.init.normal_(self.instruction.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.instruction.weight[INSTRUCTION_NONE])
        nn.init.zeros_(self.output.weight)

    def forward(self, memory, instruction_id):
        if instruction_id.ndim != 1:
            instruction_id = instruction_id.reshape(-1)

        active = (
            instruction_id.ne(INSTRUCTION_NONE)
            .to(dtype=memory.dtype)
            .unsqueeze(-1)
        )

        mem = torch.tanh(self.memory_features(memory))
        goal = torch.tanh(self.instruction(instruction_id))

        # Multiplicative composition makes BOTH remembered content and current
        # instruction necessary for the learned executive correction.
        joint = mem * goal

        residual = self.output(joint) * active
        return residual, {
            "memory_features": mem,
            "instruction_features": goal,
            "joint_features": joint,
            "active_mask": active,
        }


class ExecutiveCompositionInfant(nn.Module):
    """
    Protected admitted memory brain + newly trainable late-bound executive.
    """
    def __init__(self, admitted_brain):
        super().__init__()
        self.admitted = admitted_brain
        self.base = admitted_brain.base
        self.cfg = admitted_brain.cfg

        self.executive = LateBoundMemoryExecutive(
            memory_dim=int(self.cfg.memory_dim),
            bus_dim=int(self.cfg.bus_dim),
            instruction_count=3,
        )

    @classmethod
    def grow_from_admitted(cls, admitted_brain):
        return cls(admitted_brain)

    def initial_state(self, batch_size, device):
        return self.admitted.initial_state(batch_size, device)

    def initial_memory(self, batch_size, device):
        return self.admitted.initial_memory(batch_size, device)

    def step(
        self,
        image,
        context,
        memory,
        prev_action,
        hidden,
        instruction_id,
    ):
        admitted_out = self.admitted.step(
            image,
            context,
            memory,
            prev_action,
            hidden,
            use_memory=True,
            use_recall=True,
        )

        executive_residual, aux = self.executive(
            admitted_out["memory_in"],
            instruction_id,
        )

        executive_workspace = (
            admitted_out["workspace_for_world"]
            + executive_residual
        )
        logits = self.base.motor(executive_workspace)

        return {
            **admitted_out,
            "admitted_workspace":
                admitted_out["workspace_for_world"],
            "workspace_for_world":
                admitted_out["workspace_for_world"],
            "executive_workspace":
                executive_workspace,
            "logits": logits,
            "executive_residual":
                executive_residual,
            "executive_memory_features":
                aux["memory_features"],
            "executive_instruction_features":
                aux["instruction_features"],
            "executive_joint_features":
                aux["joint_features"],
            "executive_active_mask":
                aux["active_mask"],
        }

    def trainable_parameters(self):
        return list(self.executive.parameters())
