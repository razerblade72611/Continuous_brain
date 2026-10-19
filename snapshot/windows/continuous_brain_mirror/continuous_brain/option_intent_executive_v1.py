#!/usr/bin/env python3
from __future__ import annotations
import torch
from torch import nn
from .canonical_executive_rule_world_v3 import RULE_NONE

class OptionIntentExecutive(nn.Module):
    def __init__(self, content_dim=128, workspace_dim=128, action_dim=5,
                 feature_dim=128, intent_hidden=256, gate_hidden=128, rule_count=4):
        super().__init__()
        self.content = nn.Linear(content_dim, feature_dim, bias=False)
        self.rule = nn.Embedding(rule_count, feature_dim)
        self.intent_fusion = nn.Linear(feature_dim * 2, intent_hidden, bias=False)
        self.intent_output = nn.Linear(intent_hidden, action_dim, bias=False)
        self.gate_workspace = nn.Linear(workspace_dim, gate_hidden, bias=False)
        self.gate_action = nn.Linear(action_dim, gate_hidden, bias=False)
        self.gate_fusion = nn.Linear(gate_hidden * 2, gate_hidden, bias=False)
        self.gate_output = nn.Linear(gate_hidden, 1, bias=True)
        nn.init.xavier_uniform_(self.content.weight)
        nn.init.normal_(self.rule.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.rule.weight[RULE_NONE])
        nn.init.xavier_uniform_(self.intent_fusion.weight)
        nn.init.zeros_(self.intent_output.weight)
        nn.init.xavier_uniform_(self.gate_workspace.weight)
        nn.init.xavier_uniform_(self.gate_action.weight)
        nn.init.xavier_uniform_(self.gate_fusion.weight)
        nn.init.xavier_uniform_(self.gate_output.weight)
        nn.init.zeros_(self.gate_output.bias)

    def compute_intent(self, canonical_recall, rule_id):
        if rule_id.ndim != 1:
            rule_id = rule_id.reshape(-1)
        active = rule_id.ne(RULE_NONE).to(canonical_recall.dtype).unsqueeze(-1)
        content = torch.tanh(self.content(canonical_recall))
        rule = torch.tanh(self.rule(rule_id))
        hidden = torch.tanh(self.intent_fusion(torch.cat([content, content * rule], dim=-1)))
        raw_intent = self.intent_output(hidden) * active
        return raw_intent, {"content_features": content, "rule_features": rule,
                            "intent_hidden": hidden, "active": active}

    def compute_gate(self, historical_workspace, base_logits):
        ws = torch.tanh(self.gate_workspace(historical_workspace))
        ba = torch.tanh(self.gate_action(base_logits))
        gh = torch.tanh(self.gate_fusion(torch.cat([ws, ba], dim=-1)))
        gl = self.gate_output(gh)
        gate = torch.sigmoid(gl)
        return gate, gl, {"gate_workspace_features": ws,
                          "gate_action_features": ba, "gate_hidden": gh}

    def forward(self, historical_workspace, canonical_recall, rule_id, base_logits):
        raw_intent, ia = self.compute_intent(canonical_recall, rule_id)
        gate, gate_logit, ga = self.compute_gate(historical_workspace, base_logits)
        delta = raw_intent * gate
        return delta, {**ia, **ga, "raw_intent": raw_intent,
                       "gate": gate, "gate_logit": gate_logit}

class OptionIntentExecutiveInfant(nn.Module):
    def __init__(self, admitted):
        super().__init__()
        self.admitted = admitted
        self.base = admitted.base
        self.cfg = admitted.cfg
        self.executive = OptionIntentExecutive(
            content_dim=int(self.cfg.bus_dim),
            workspace_dim=int(self.cfg.bus_dim),
            action_dim=5, feature_dim=128, intent_hidden=256,
            gate_hidden=128, rule_count=4)

    @classmethod
    def grow_from_admitted(cls, admitted):
        return cls(admitted)

    def initial_state(self, batch_size, device):
        return self.admitted.initial_state(batch_size, device)

    def initial_memory(self, batch_size, device):
        return self.admitted.initial_memory(batch_size, device)

    def components(self, image, context, memory, prev_action, hidden):
        hist = self.admitted.historical_step(image, context, memory, prev_action, hidden)
        recall_residual, recall_aux = self.admitted.recall(hist["hidden"], memory)
        admitted_workspace = hist["historical_workspace"] + recall_residual
        base_logits = self.base.motor(admitted_workspace)
        return hist, recall_residual, recall_aux, admitted_workspace, base_logits

    def compose(self, historical_workspace, canonical_recall, rule_id, base_logits):
        delta, aux = self.executive(historical_workspace, canonical_recall, rule_id, base_logits)
        return base_logits + delta, delta, aux

    def step(self, image, context, memory, prev_action, hidden, rule_id):
        hist, recall_residual, recall_aux, admitted_workspace, base_logits = self.components(
            image, context, memory, prev_action, hidden)
        logits, delta, aux = self.compose(
            hist["historical_workspace"], recall_residual, rule_id, base_logits)
        next_memory, mem_aux = self.admitted.memory(hist["hidden"], memory)
        return {
            **hist,
            "workspace_for_world": admitted_workspace,
            "admitted_workspace": admitted_workspace,
            "logits": logits,
            "base_logits": base_logits,
            "executive_intent_delta": delta,
            "executive_gate": aux["gate"],
            "executive_gate_logit": aux["gate_logit"],
            "raw_intent": aux["raw_intent"],
            "memory_in": memory,
            "memory_next": next_memory,
            "memory_candidate": mem_aux["candidate"],
            "memory_write_gate": mem_aux["write_gate"],
            "recall_residual": recall_residual,
            "recall_decoded": recall_aux["decoded"],
        }

    def trainable_parameters(self):
        return list(self.executive.parameters())
