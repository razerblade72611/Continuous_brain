#!/usr/bin/env python3
"""
Canonical recalled-content executive.

Architectural lesson from v0.13.2 route-coherence diagnostics:
the admitted direct recall route D and executive route E should not independently
read the raw 64-D memory.  They must share one recalled-content representation.

Protected path:
    raw memory64
        -> admitted nonlinear recall (frozen)
        -> canonical recalled content128
             ├─ added to historical workspace exactly as admitted
             └─ consumed by late-rule executive

The new executive NEVER rereads memory64.

Active rule:
    tanh(canonical_content128) * tanh(rule_embedding128)
        -> zero-init Linear128->128
        -> additive workspace delta

RULE_NONE is a hard exact no-op.

The admitted writer/reader/base are frozen and excluded from the optimizer.
"""
from __future__ import annotations

import torch
from torch import nn

from .canonical_executive_rule_world_v3 import RULE_NONE


class CanonicalRecallRuleExecutive(nn.Module):
    def __init__(self, bus_dim=128, rule_count=4):
        super().__init__()
        self.bus_dim = int(bus_dim)
        self.rule = nn.Embedding(int(rule_count), self.bus_dim)
        self.output = nn.Linear(self.bus_dim, self.bus_dim, bias=False)

        nn.init.normal_(self.rule.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.rule.weight[RULE_NONE])
        nn.init.zeros_(self.output.weight)

    def forward(self, canonical_recall, rule_id):
        if rule_id.ndim != 1:
            rule_id = rule_id.reshape(-1)
        active = rule_id.ne(RULE_NONE).to(canonical_recall.dtype).unsqueeze(-1)
        content = torch.tanh(canonical_recall)
        rule = torch.tanh(self.rule(rule_id))
        joint = content * rule
        delta = self.output(joint) * active
        return delta, {
            "canonical_content": content,
            "rule_features": rule,
            "joint_features": joint,
            "active": active,
        }


class CanonicalRecallExecutiveInfant(nn.Module):
    def __init__(self, admitted):
        super().__init__()
        self.admitted = admitted
        self.base = admitted.base
        self.cfg = admitted.cfg
        self.executive = CanonicalRecallRuleExecutive(
            bus_dim=int(self.cfg.bus_dim),
            rule_count=4,
        )

    @classmethod
    def grow_from_admitted(cls, admitted):
        return cls(admitted)

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
        rule_id,
    ):
        # Reconstruct the admitted step explicitly so the SAME recall residual
        # is both (a) added to workspace and (b) used as executive content.
        hist = self.admitted.historical_step(
            image,
            context,
            memory,
            prev_action,
            hidden,
        )

        recall_residual, recall_aux = self.admitted.recall(
            hist["hidden"],
            memory,
        )

        admitted_workspace = (
            hist["historical_workspace"]
            + recall_residual
        )

        executive_delta, aux = self.executive(
            recall_residual,
            rule_id,
        )
        executive_workspace = (
            admitted_workspace
            + executive_delta
        )
        logits = self.base.motor(executive_workspace)

        next_memory, mem_aux = self.admitted.memory(
            hist["hidden"],
            memory,
        )

        return {
            **hist,
            "workspace_for_world": admitted_workspace,
            "admitted_workspace": admitted_workspace,
            "executive_workspace": executive_workspace,
            "logits": logits,
            "memory_in": memory,
            "memory_next": next_memory,
            "memory_candidate": mem_aux["candidate"],
            "memory_write_gate": mem_aux["write_gate"],
            "recall_residual": recall_residual,
            "recall_decoded": recall_aux["decoded"],
            "executive_delta": executive_delta,
            "canonical_content": aux["canonical_content"],
            "executive_rule_features": aux["rule_features"],
            "executive_joint_features": aux["joint_features"],
        }

    def trainable_parameters(self):
        return list(self.executive.parameters())
