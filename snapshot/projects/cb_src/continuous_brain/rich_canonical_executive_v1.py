#!/usr/bin/env python3
"""
Rich canonical recalled-content executive v1.

The admitted writer/recall/base are frozen. The executive never reads raw
memory64. It consumes the same canonical recall residual already added to the
admitted workspace.

Architecture:
  canonical_recall128 -> trainable content projection 128->128 -> tanh
  rule_id             -> embedding128                         -> tanh
  concat(content, rule, content*rule) [384]
      -> fusion 384->256 -> tanh
      -> zero-init output 256->128
      -> workspace delta

RULE_NONE is hard-masked to an exact zero delta.
"""
from __future__ import annotations

import torch
from torch import nn

from .canonical_executive_rule_world_v3 import RULE_NONE


class RichCanonicalRuleExecutive(nn.Module):
    def __init__(
        self,
        bus_dim: int = 128,
        hidden_dim: int = 256,
        rule_count: int = 4,
    ):
        super().__init__()
        self.bus_dim = int(bus_dim)
        self.hidden_dim = int(hidden_dim)
        self.rule_count = int(rule_count)

        self.content = nn.Linear(
            self.bus_dim,
            self.bus_dim,
            bias=False,
        )
        self.rule = nn.Embedding(
            self.rule_count,
            self.bus_dim,
        )
        self.fusion = nn.Linear(
            self.bus_dim * 3,
            self.hidden_dim,
            bias=False,
        )
        self.output = nn.Linear(
            self.hidden_dim,
            self.bus_dim,
            bias=False,
        )

        nn.init.xavier_uniform_(self.content.weight)
        nn.init.normal_(self.rule.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.rule.weight[RULE_NONE])
        nn.init.xavier_uniform_(self.fusion.weight)
        nn.init.zeros_(self.output.weight)

    def forward(self, canonical_recall, rule_id):
        if rule_id.ndim != 1:
            rule_id = rule_id.reshape(-1)

        active = (
            rule_id.ne(RULE_NONE)
            .to(canonical_recall.dtype)
            .unsqueeze(-1)
        )

        content = torch.tanh(
            self.content(canonical_recall)
        )
        rule = torch.tanh(
            self.rule(rule_id)
        )
        interaction = content * rule
        fused = torch.cat(
            [content, rule, interaction],
            dim=-1,
        )
        hidden = torch.tanh(
            self.fusion(fused)
        )
        delta = self.output(hidden) * active

        return delta, {
            "content_features": content,
            "rule_features": rule,
            "interaction_features": interaction,
            "fusion_hidden": hidden,
            "active": active,
        }


class RichCanonicalExecutiveInfant(nn.Module):
    def __init__(self, admitted):
        super().__init__()
        self.admitted = admitted
        self.base = admitted.base
        self.cfg = admitted.cfg

        self.executive = RichCanonicalRuleExecutive(
            bus_dim=int(self.cfg.bus_dim),
            hidden_dim=256,
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

        executive_delta, exec_aux = self.executive(
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
            "executive_content_features":
                exec_aux["content_features"],
            "executive_rule_features":
                exec_aux["rule_features"],
            "executive_interaction_features":
                exec_aux["interaction_features"],
            "executive_fusion_hidden":
                exec_aux["fusion_hidden"],
        }

    def trainable_parameters(self):
        return list(self.executive.parameters())
