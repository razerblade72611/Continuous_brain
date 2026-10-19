#!/usr/bin/env python3
"""
Additive differentiable working-memory wrapper for the protected InfantBrain.

This module does not alter InfantBrain or its checkpoint.  The historical base
is passed in and can remain completely frozen.  The only trainable state owned
here is a 64-dimensional gated working-memory writer.

At construction the candidate projection is exactly zero, so with an initial
zero memory state the enabled wrapper is an exact behavioral no-op for any
number of steps.  Task loss can nevertheless reach the candidate projection
through later uses of memory under full BPTT.
"""

import torch
from torch import nn


class GatedWorkingMemory(nn.Module):
    def __init__(self, nucleus_dim: int, memory_dim: int):
        super().__init__()
        self.nucleus_dim = int(nucleus_dim)
        self.memory_dim = int(memory_dim)

        self.candidate = nn.Linear(self.nucleus_dim, self.memory_dim)
        self.write_gate = nn.Linear(
            self.nucleus_dim + self.memory_dim,
            self.memory_dim,
        )

        # Exact no-op initialization:
        # candidate(hidden) == 0 exactly, therefore m_{t+1} == 0 whenever
        # m_t == 0, independent of the gate value.
        nn.init.zeros_(self.candidate.weight)
        nn.init.zeros_(self.candidate.bias)

        # Neutral 0.5 gate initially.  It receives no gradient on the very
        # first update because candidate==memory==0; once candidate moves away
        # from zero it becomes trainable through the same task loss.
        nn.init.zeros_(self.write_gate.weight)
        nn.init.zeros_(self.write_gate.bias)

    def initial_state(self, batch_size: int, device):
        return torch.zeros(
            int(batch_size),
            self.memory_dim,
            device=device,
        )

    def forward(self, hidden, memory):
        if hidden.ndim != 2 or hidden.shape[-1] != self.nucleus_dim:
            raise ValueError(
                f"hidden shape must be [B,{self.nucleus_dim}], "
                f"got {tuple(hidden.shape)}"
            )
        if memory.ndim != 2 or memory.shape[-1] != self.memory_dim:
            raise ValueError(
                f"memory shape must be [B,{self.memory_dim}], "
                f"got {tuple(memory.shape)}"
            )
        if hidden.shape[0] != memory.shape[0]:
            raise ValueError("hidden/memory batch mismatch")

        candidate = torch.tanh(self.candidate(hidden))
        gate_input = torch.cat((hidden, memory), dim=-1)
        write_gate = torch.sigmoid(self.write_gate(gate_input))
        next_memory = memory + write_gate * (candidate - memory)

        return next_memory, {
            "candidate": candidate,
            "write_gate": write_gate,
        }


class WorkingMemoryInfant(nn.Module):
    """
    Protected wrapper around a trained InfantBrain.

    Forward path uses the base model's real vision, CorticalNucleus projected
    streams, recurrent core, workspace/growth path, and motor head.  The only
    new mechanism is a differentiable working-memory state that is written
    after each base step and supplied to the base memory stream on the next
    step.
    """

    def __init__(self, base, cfg):
        super().__init__()
        self.base = base
        self.cfg = cfg

        nucleus_dim = int(cfg.nucleus_dim)
        memory_dim = int(cfg.memory_dim)
        self.memory = GatedWorkingMemory(nucleus_dim, memory_dim)

    @classmethod
    def grow_from(cls, base, cfg):
        return cls(base, cfg)

    def initial_state(self, batch_size, device):
        return self.base.nucleus.initial_state(batch_size, device)

    def initial_memory(self, batch_size, device):
        return self.memory.initial_state(batch_size, device)

    def _workspace_from_hidden(self, hidden):
        # Same path established by the qualified competitive wrapper/probes.
        return self.base.nucleus.growth(
            self.base.nucleus.workspace(hidden)
        )

    def base_step(self, image, context, memory, prev_action, hidden):
        semantic = self.base.vision(image)
        spatial = self.base.spatial(image)
        fused = self.base.fuse_vision(semantic, spatial)

        streams = self.base.nucleus.project_streams(
            fused,
            context,
            memory,
            prev_action,
        )
        core_input = torch.cat(tuple(streams), dim=-1)
        next_hidden = self.base.nucleus.core(core_input, hidden)
        workspace = self._workspace_from_hidden(next_hidden)
        logits = self.base.motor(workspace)

        return {
            "hidden": next_hidden,
            "workspace_for_world": workspace,
            "logits": logits,
            "streams": streams,
        }

    def step(
        self,
        image,
        context,
        memory,
        prev_action,
        hidden,
        *,
        use_memory=True,
    ):
        if not use_memory:
            memory_in = torch.zeros_like(memory)
        else:
            memory_in = memory

        out = self.base_step(
            image,
            context,
            memory_in,
            prev_action,
            hidden,
        )

        if use_memory:
            next_memory, mem_aux = self.memory(
                out["hidden"],
                memory,
            )
        else:
            next_memory = torch.zeros_like(memory)
            mem_aux = {
                "candidate": torch.zeros_like(memory),
                "write_gate": torch.zeros_like(memory),
            }

        return {
            **out,
            "memory_in": memory_in,
            "memory_next": next_memory,
            "memory_candidate": mem_aux["candidate"],
            "memory_write_gate": mem_aux["write_gate"],
        }

    def predict_world(self, workspace, action_one_hot, base_context):
        # Delegate to the historical world model.  This wrapper changes only
        # the memory path.
        return self.base.world(
            workspace,
            action_one_hot,
            base_context=base_context,
        )

    def trainable_parameters(self):
        return list(self.memory.parameters())
