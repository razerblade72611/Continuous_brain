#!/usr/bin/env python3
"""
Novel-rule late-bound executive compositor v2.

Unlike v1, this module has no NORMAL instruction.  All three active rules are
novel relative to the admitted brain's inherited LEFT/RIGHT policy, so every
successful rule must use the new executive pathway.

The admitted brain remains frozen/protected.
"""
from __future__ import annotations

import torch
from torch import nn

from .executive_novel_rule_world_v2 import RULE_NONE


class NovelRuleExecutive(nn.Module):
    def __init__(self, memory_dim=64, bus_dim=128, rule_count=4):
        super().__init__()
        self.memory_features = nn.Linear(memory_dim, bus_dim, bias=False)
        self.rule = nn.Embedding(rule_count, bus_dim)
        self.output = nn.Linear(bus_dim, bus_dim, bias=False)

        nn.init.xavier_uniform_(self.memory_features.weight)
        nn.init.normal_(self.rule.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.rule.weight[RULE_NONE])
        nn.init.zeros_(self.output.weight)

    def forward(self, memory, rule_id):
        if rule_id.ndim != 1:
            rule_id = rule_id.reshape(-1)

        active = rule_id.ne(RULE_NONE).to(memory.dtype).unsqueeze(-1)
        mem = torch.tanh(self.memory_features(memory))
        rule = torch.tanh(self.rule(rule_id))
        joint = mem * rule
        residual = self.output(joint) * active
        return residual, {
            "memory_features": mem,
            "rule_features": rule,
            "joint_features": joint,
            "active": active,
        }


class NovelRuleExecutiveInfant(nn.Module):
    def __init__(self, admitted):
        super().__init__()
        self.admitted = admitted
        self.base = admitted.base
        self.cfg = admitted.cfg
        self.executive = NovelRuleExecutive(
            memory_dim=int(self.cfg.memory_dim),
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
        admitted = self.admitted.step(
            image,
            context,
            memory,
            prev_action,
            hidden,
            use_memory=True,
            use_recall=True,
        )

        residual, aux = self.executive(
            admitted["memory_in"],
            rule_id,
        )
        workspace = admitted["workspace_for_world"] + residual
        logits = self.base.motor(workspace)

        return {
            **admitted,
            "logits": logits,
            "admitted_workspace": admitted["workspace_for_world"],
            "executive_workspace": workspace,
            "executive_residual": residual,
            "executive_memory_features": aux["memory_features"],
            "executive_rule_features": aux["rule_features"],
            "executive_joint_features": aux["joint_features"],
        }

    def trainable_parameters(self):
        return list(self.executive.parameters())
