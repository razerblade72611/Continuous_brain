#!/usr/bin/env python3
"""
Contextual canonical recalled-content executive v1.

The executive receives:
  H = frozen historical workspace context128
  D = frozen canonical recall content128
  R = late rule embedding128

It never receives:
  raw memory64,
  image,
  raw context vector,
  recurrent hidden state.

Every executive feature is content-dependent. If D is exactly zero, every
feature path is exactly zero and the executive delta is exactly zero.

Features:
  C  = tanh(Wc D)
  X  = tanh(Wx H)
  R  = tanh(Erule)
  [C, C*R, C*X, C*R*X] -> Linear512->256 -> tanh
                             -> zero-init Linear256->128
                             -> workspace delta
"""
from __future__ import annotations

import torch
from torch import nn

from .canonical_executive_rule_world_v3 import RULE_NONE


class ContextualCanonicalRuleExecutive(nn.Module):
    def __init__(self, bus_dim: int = 128, hidden_dim: int = 256, rule_count: int = 4):
        super().__init__()
        self.bus_dim = int(bus_dim)
        self.hidden_dim = int(hidden_dim)

        self.content = nn.Linear(bus_dim, bus_dim, bias=False)
        self.context = nn.Linear(bus_dim, bus_dim, bias=False)
        self.rule = nn.Embedding(rule_count, bus_dim)
        self.fusion = nn.Linear(bus_dim * 4, hidden_dim, bias=False)
        self.output = nn.Linear(hidden_dim, bus_dim, bias=False)

        nn.init.xavier_uniform_(self.content.weight)
        nn.init.xavier_uniform_(self.context.weight)
        nn.init.normal_(self.rule.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.rule.weight[RULE_NONE])
        nn.init.xavier_uniform_(self.fusion.weight)
        nn.init.zeros_(self.output.weight)

    def forward(self, historical_workspace, canonical_recall, rule_id):
        if rule_id.ndim != 1:
            rule_id = rule_id.reshape(-1)

        active = rule_id.ne(RULE_NONE).to(canonical_recall.dtype).unsqueeze(-1)

        content = torch.tanh(self.content(canonical_recall))
        context = torch.tanh(self.context(historical_workspace))
        rule = torch.tanh(self.rule(rule_id))

        cr = content * rule
        cx = content * context
        crx = content * rule * context

        fused = torch.cat([content, cr, cx, crx], dim=-1)
        hidden = torch.tanh(self.fusion(fused))
        delta = self.output(hidden) * active

        return delta, {
            "content_features": content,
            "context_features": context,
            "rule_features": rule,
            "fusion_hidden": hidden,
            "active": active,
        }


class ContextualCanonicalExecutiveInfant(nn.Module):
    def __init__(self, admitted):
        super().__init__()
        self.admitted = admitted
        self.base = admitted.base
        self.cfg = admitted.cfg
        self.executive = ContextualCanonicalRuleExecutive(
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

    def components(self, image, context, memory, prev_action, hidden):
        hist = self.admitted.historical_step(
            image, context, memory, prev_action, hidden
        )
        recall_residual, recall_aux = self.admitted.recall(
            hist["hidden"], memory
        )
        return hist, recall_residual, recall_aux

    def compose(self, hist, canonical_recall, rule_id):
        delta, aux = self.executive(
            hist["historical_workspace"],
            canonical_recall,
            rule_id,
        )
        admitted_workspace = hist["historical_workspace"] + canonical_recall
        executive_workspace = admitted_workspace + delta
        logits = self.base.motor(executive_workspace)
        return logits, delta, aux, admitted_workspace, executive_workspace

    def step(self, image, context, memory, prev_action, hidden, rule_id):
        hist, recall_residual, recall_aux = self.components(
            image, context, memory, prev_action, hidden
        )
        logits, delta, exec_aux, admitted_workspace, executive_workspace = self.compose(
            hist, recall_residual, rule_id
        )
        next_memory, mem_aux = self.admitted.memory(hist["hidden"], memory)

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
            "executive_delta": delta,
            "executive_content_features": exec_aux["content_features"],
            "executive_context_features": exec_aux["context_features"],
            "executive_rule_features": exec_aux["rule_features"],
            "executive_fusion_hidden": exec_aux["fusion_hidden"],
        }

    def trainable_parameters(self):
        return list(self.executive.parameters())
