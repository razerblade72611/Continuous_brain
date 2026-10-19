#!/usr/bin/env python3
"""
Controlled working-memory recall arms for the protected Continuous Brain child.

M1.5 / "linear":
    residual_t = P_0(m_t)

M2 / "gated":
    decoded_t  = P_0(m_t)
    gate_t     = sigmoid(G([h_t, m_t]))
    residual_t = gate_t * decoded_t

P_0 is a bias-free, exactly zero-initialized memory->workspace projection.
Therefore both arms are exact behavioral no-ops at construction for arbitrary
inputs, not merely when memory happens to be zero.

The historical writer and historical memory route remain present.  The only new
difference from Working Memory M1 is an additive recall residual at the shared
workspace, before the frozen motor/world consumers.

No historical base tensor is owned by these modules.
"""

from __future__ import annotations

import torch
from torch import nn

from continuous_brain.working_memory_infant import GatedWorkingMemory


class LinearRecallBridge(nn.Module):
    """M1.5: bare bias-free zero-init memory -> workspace residual."""

    def __init__(self, memory_dim: int, bus_dim: int):
        super().__init__()
        self.memory_dim = int(memory_dim)
        self.bus_dim = int(bus_dim)
        self.projection = nn.Linear(
            self.memory_dim,
            self.bus_dim,
            bias=False,
        )
        nn.init.zeros_(self.projection.weight)

    def forward(self, hidden, memory):
        del hidden
        residual = self.projection(memory)
        return residual, {
            "decoded": residual,
            "recall_gate": None,
        }


class GatedRecallBridge(nn.Module):
    """
    M2: the identical zero-init projection used by M1.5, plus a contextual gate.

    The gate cannot create a memory-independent shortcut because it only
    multiplies decoded memory.  When memory is zero, decoded memory is zero;
    when the zero-initialized projection has not learned yet, the gate is also
    behaviorally inert.
    """

    def __init__(
        self,
        nucleus_dim: int,
        memory_dim: int,
        bus_dim: int,
    ):
        super().__init__()
        self.nucleus_dim = int(nucleus_dim)
        self.memory_dim = int(memory_dim)
        self.bus_dim = int(bus_dim)

        self.projection = nn.Linear(
            self.memory_dim,
            self.bus_dim,
            bias=False,
        )
        nn.init.zeros_(self.projection.weight)

        self.gate = nn.Linear(
            self.nucleus_dim + self.memory_dim,
            self.bus_dim,
            bias=True,
        )
        # Neutral 0.5 gate.  It should receive exactly zero gradient while the
        # projection output is exactly zero, then become load-bearing after the
        # projection moves.
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, hidden, memory):
        decoded = self.projection(memory)
        gate_input = torch.cat((hidden, memory), dim=-1)
        recall_gate = torch.sigmoid(self.gate(gate_input))
        residual = recall_gate * decoded
        return residual, {
            "decoded": decoded,
            "recall_gate": recall_gate,
        }


class RecallBridgeWorkingMemoryInfant(nn.Module):
    """
    Protected writer + controlled direct workspace recall arm.

    The historical memory route remains unchanged:
        m_t -> frozen memory projection -> frozen recurrent core

    The experimental direct recall route is additive:
        m_t -> recall bridge -> + historical workspace

    The writer still writes m_(t+1) only after the current recurrent step.
    """

    VALID_ARMS = ("linear", "gated")

    def __init__(self, base, cfg, *, arm: str):
        super().__init__()
        if arm not in self.VALID_ARMS:
            raise ValueError(
                f"arm must be one of {self.VALID_ARMS}, got {arm!r}"
            )

        self.base = base
        self.cfg = cfg
        self.arm = arm

        nucleus_dim = int(cfg.nucleus_dim)
        memory_dim = int(cfg.memory_dim)
        bus_dim = int(cfg.bus_dim)

        self.memory = GatedWorkingMemory(
            nucleus_dim,
            memory_dim,
        )

        if arm == "linear":
            self.recall = LinearRecallBridge(
                memory_dim,
                bus_dim,
            )
        else:
            self.recall = GatedRecallBridge(
                nucleus_dim,
                memory_dim,
                bus_dim,
            )

    @classmethod
    def grow_from(cls, base, cfg, *, arm: str):
        return cls(base, cfg, arm=arm)

    def initial_state(self, batch_size, device):
        return self.base.nucleus.initial_state(
            batch_size, device
        )

    def initial_memory(self, batch_size, device):
        return self.memory.initial_state(
            batch_size, device
        )

    def _historical_workspace(self, hidden):
        return self.base.nucleus.growth(
            self.base.nucleus.workspace(hidden)
        )

    def historical_step(
        self,
        image,
        context,
        memory,
        prev_action,
        hidden,
    ):
        semantic = self.base.vision(image)
        spatial = self.base.spatial(image)
        fused = self.base.fuse_vision(
            semantic, spatial
        )
        streams = self.base.nucleus.project_streams(
            fused,
            context,
            memory,
            prev_action,
        )
        core_input = torch.cat(
            tuple(streams), dim=-1
        )
        next_hidden = self.base.nucleus.core(
            core_input, hidden
        )
        historical_workspace = (
            self._historical_workspace(next_hidden)
        )
        historical_logits = self.base.motor(
            historical_workspace
        )
        return {
            "hidden": next_hidden,
            "historical_workspace":
                historical_workspace,
            "historical_logits":
                historical_logits,
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
        use_recall=True,
    ):
        memory_in = (
            memory
            if use_memory
            else torch.zeros_like(memory)
        )

        out = self.historical_step(
            image,
            context,
            memory_in,
            prev_action,
            hidden,
        )

        if use_recall and use_memory:
            residual, recall_aux = self.recall(
                out["hidden"],
                memory_in,
            )
        else:
            residual = torch.zeros_like(
                out["historical_workspace"]
            )
            recall_aux = {
                "decoded": torch.zeros_like(
                    out["historical_workspace"]
                ),
                "recall_gate": None,
            }

        workspace = (
            out["historical_workspace"]
            + residual
        )
        logits = self.base.motor(workspace)

        if use_memory:
            next_memory, mem_aux = self.memory(
                out["hidden"],
                memory,
            )
        else:
            next_memory = torch.zeros_like(memory)
            mem_aux = {
                "candidate": torch.zeros_like(
                    memory
                ),
                "write_gate": torch.zeros_like(
                    memory
                ),
            }

        return {
            **out,
            "workspace_for_world": workspace,
            "logits": logits,
            "memory_in": memory_in,
            "memory_next": next_memory,
            "memory_candidate":
                mem_aux["candidate"],
            "memory_write_gate":
                mem_aux["write_gate"],
            "recall_residual": residual,
            "recall_decoded":
                recall_aux["decoded"],
            "recall_gate":
                recall_aux["recall_gate"],
        }

    def predict_world(
        self,
        workspace,
        action_one_hot,
        base_context,
    ):
        return self.base.world(
            workspace,
            action_one_hot,
            base_context=base_context,
        )

    def trainable_parameters(self):
        return list(self.memory.parameters()) + list(
            self.recall.parameters()
        )
