#!/usr/bin/env python3
"""
Executive Action Intent Cortex v1.

This is a fundamental control-interface redesign after workspace-delta
executives failed to master TRAIN despite intact remembered content.

Protected lower brain:
  historical workspace + canonical recall -> frozen motor -> base logits5

Executive inputs:
  D = canonical recall128
  R = late rule
  B = frozen base motor logits5

Executive output:
  action-logit residual5

Final action logits:
  base_logits5 + intent_delta5

The executive does NOT alter:
- recurrent hidden state;
- admitted workspace;
- world-model input;
- canonical recall;
- frozen motor parameters.

Every executive feature is content-dependent. Zero canonical recall forces
intent_delta to exactly zero. RULE_NONE is also an exact no-op.

This establishes a clean hierarchy:
  memory/representation -> executive action intent -> protected motor baseline.
"""
from __future__ import annotations

import torch
from torch import nn

from .canonical_executive_rule_world_v3 import RULE_NONE


class ActionIntentExecutive(nn.Module):
    def __init__(
        self,
        content_dim: int = 128,
        action_dim: int = 5,
        feature_dim: int = 128,
        hidden_dim: int = 256,
        rule_count: int = 4,
    ):
        super().__init__()
        self.content = nn.Linear(content_dim, feature_dim, bias=False)
        self.rule = nn.Embedding(rule_count, feature_dim)
        self.base_action = nn.Linear(action_dim, feature_dim, bias=False)
        self.fusion = nn.Linear(feature_dim * 4, hidden_dim, bias=False)
        self.output = nn.Linear(hidden_dim, action_dim, bias=False)

        nn.init.xavier_uniform_(self.content.weight)
        nn.init.normal_(self.rule.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.rule.weight[RULE_NONE])
        nn.init.xavier_uniform_(self.base_action.weight)
        nn.init.xavier_uniform_(self.fusion.weight)
        nn.init.zeros_(self.output.weight)

    def forward(self, canonical_recall, rule_id, base_logits):
        if rule_id.ndim != 1:
            rule_id = rule_id.reshape(-1)

        active = (
            rule_id.ne(RULE_NONE)
            .to(canonical_recall.dtype)
            .unsqueeze(-1)
        )

        content = torch.tanh(self.content(canonical_recall))
        rule = torch.tanh(self.rule(rule_id))
        base = torch.tanh(self.base_action(base_logits))

        cr = content * rule
        cb = content * base
        crb = content * rule * base

        fused = torch.cat([content, cr, cb, crb], dim=-1)
        hidden = torch.tanh(self.fusion(fused))
        delta = self.output(hidden) * active

        return delta, {
            "content_features": content,
            "rule_features": rule,
            "base_action_features": base,
            "fusion_hidden": hidden,
            "active": active,
        }


class ActionIntentExecutiveInfant(nn.Module):
    def __init__(self, admitted):
        super().__init__()
        self.admitted = admitted
        self.base = admitted.base
        self.cfg = admitted.cfg

        self.executive = ActionIntentExecutive(
            content_dim=int(self.cfg.bus_dim),
            action_dim=5,
            feature_dim=128,
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

    def components(self, image, context, memory, prev_action, hidden):
        hist = self.admitted.historical_step(
            image, context, memory, prev_action, hidden
        )
        recall_residual, recall_aux = self.admitted.recall(
            hist["hidden"], memory
        )
        admitted_workspace = (
            hist["historical_workspace"] + recall_residual
        )
        base_logits = self.base.motor(admitted_workspace)
        return hist, recall_residual, recall_aux, admitted_workspace, base_logits

    def compose(self, canonical_recall, rule_id, base_logits):
        delta, aux = self.executive(
            canonical_recall,
            rule_id,
            base_logits,
        )
        logits = base_logits + delta
        return logits, delta, aux

    def step(self, image, context, memory, prev_action, hidden, rule_id):
        (
            hist,
            recall_residual,
            recall_aux,
            admitted_workspace,
            base_logits,
        ) = self.components(
            image, context, memory, prev_action, hidden
        )

        logits, delta, exec_aux = self.compose(
            recall_residual,
            rule_id,
            base_logits,
        )

        next_memory, mem_aux = self.admitted.memory(
            hist["hidden"], memory
        )

        return {
            **hist,
            "workspace_for_world": admitted_workspace,
            "admitted_workspace": admitted_workspace,
            "logits": logits,
            "base_logits": base_logits,
            "executive_intent_delta": delta,
            "memory_in": memory,
            "memory_next": next_memory,
            "memory_candidate": mem_aux["candidate"],
            "memory_write_gate": mem_aux["write_gate"],
            "recall_residual": recall_residual,
            "recall_decoded": recall_aux["decoded"],
            "intent_content_features": exec_aux["content_features"],
            "intent_rule_features": exec_aux["rule_features"],
            "intent_base_action_features": exec_aux["base_action_features"],
            "intent_fusion_hidden": exec_aux["fusion_hidden"],
        }

    def trainable_parameters(self):
        return list(self.executive.parameters())
