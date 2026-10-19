#!/usr/bin/env python3
"""
Continuous Brain v0.19.0 — Recurrent Option Progress Controller.

Scientific purpose
------------------
v0.18.1 showed that the response boundary is NOT statically observable enough
from workspace/base logits, hidden, or their combinations.  However the v0.18
WHAT pathway is already strong (~99% response/donor accuracy).

Therefore this module freezes the learned v0.18 WHAT/action-intent head and
adds a recurrent WHEN/progress state.

WHAT (frozen parent):
    canonical recall128 + late rule -> raw action intent5

WHEN (new trainable):
    protected post-core hidden
    + historical workspace
    + frozen base logits
    + previous option-progress state
        -> recurrent option-progress state128
        -> scalar execution gate

Final:
    logits = base_logits
             + intent_gain * execution_gate * frozen_raw_intent

Properties
----------
- admitted memory/recall/base remain frozen;
- v0.18 WHAT weights are frozen;
- no raw memory64 reaches the progress controller;
- progress controller never receives rule identity;
- RULE_NONE resets option state and produces exact zero executive delta;
- zero canonical recall produces exact zero intent delta;
- world-model workspace is unchanged;
- gate is recurrent, so timing can be represented as temporal phase instead of
  requiring a single static state to expose the response boundary.
"""
from __future__ import annotations

import torch
from torch import nn

from .canonical_executive_rule_world_v3 import RULE_NONE
from .option_intent_executive_v1 import OptionIntentExecutive


class RecurrentOptionProgressGate(nn.Module):
    def __init__(
        self,
        protected_hidden_dim: int,
        workspace_dim: int = 128,
        action_dim: int = 5,
        progress_dim: int = 128,
    ):
        super().__init__()
        self.progress_dim = int(progress_dim)

        self.hidden_proj = nn.Linear(
            int(protected_hidden_dim),
            128,
            bias=False,
        )
        self.workspace_proj = nn.Linear(
            int(workspace_dim),
            64,
            bias=False,
        )
        self.action_proj = nn.Linear(
            int(action_dim),
            32,
            bias=False,
        )
        self.input_fusion = nn.Linear(
            128 + 64 + 32,
            self.progress_dim,
            bias=False,
        )
        self.recurrent = nn.Linear(
            self.progress_dim,
            self.progress_dim,
            bias=False,
        )
        self.gate_output = nn.Linear(
            self.progress_dim,
            1,
            bias=True,
        )

        nn.init.xavier_uniform_(self.hidden_proj.weight)
        nn.init.xavier_uniform_(self.workspace_proj.weight)
        nn.init.xavier_uniform_(self.action_proj.weight)
        nn.init.xavier_uniform_(self.input_fusion.weight)
        nn.init.orthogonal_(self.recurrent.weight)

        # Exact zero gate at growth.  We use a centered sigmoid:
        # 2*(sigmoid(0)-0.5) == 0 exactly.
        nn.init.zeros_(self.gate_output.weight)
        nn.init.zeros_(self.gate_output.bias)

    def initial_state(self, batch_size: int, device):
        return torch.zeros(
            int(batch_size),
            self.progress_dim,
            device=device,
        )

    def forward(
        self,
        protected_hidden,
        historical_workspace,
        base_logits,
        option_state,
        active_mask,
    ):
        if active_mask.ndim == 1:
            active_mask = active_mask.unsqueeze(-1)
        active_mask = active_mask.to(protected_hidden.dtype)

        h = torch.tanh(self.hidden_proj(protected_hidden))
        w = torch.tanh(self.workspace_proj(historical_workspace))
        a = torch.tanh(self.action_proj(base_logits))
        inp = torch.tanh(
            self.input_fusion(
                torch.cat([h, w, a], dim=-1)
            )
        )

        candidate = torch.tanh(
            inp + self.recurrent(option_state)
        )

        # Before an option begins, progress state is exactly reset.
        next_state = candidate * active_mask

        gate_logit = self.gate_output(next_state)
        gate = 2.0 * (
            torch.sigmoid(gate_logit) - 0.5
        )
        gate = gate * active_mask

        return gate, gate_logit, next_state, {
            "protected_hidden_features": h,
            "workspace_features": w,
            "action_features": a,
            "progress_input": inp,
        }


class RecurrentOptionProgressInfant(nn.Module):
    def __init__(self, admitted):
        super().__init__()
        self.admitted = admitted
        self.base = admitted.base
        self.cfg = admitted.cfg

        hidden_dim = int(
            admitted.memory.candidate.in_features
        )
        workspace_dim = int(self.cfg.bus_dim)

        # Same class/shape as the v0.18 parent.  Parent weights are loaded at
        # runtime, then this entire module is frozen.  Its old static gate is
        # retained only so the parent state_dict loads exactly; it is never used.
        self.frozen_parent_executive = OptionIntentExecutive(
            content_dim=workspace_dim,
            workspace_dim=workspace_dim,
            action_dim=5,
            feature_dim=128,
            intent_hidden=256,
            gate_hidden=128,
            rule_count=4,
        )
        for p in self.frozen_parent_executive.parameters():
            p.requires_grad_(False)

        self.progress = RecurrentOptionProgressGate(
            protected_hidden_dim=hidden_dim,
            workspace_dim=workspace_dim,
            action_dim=5,
            progress_dim=128,
        )

        # Recalibrates the already-learned raw intent to a gate whose target is
        # now 1 at execution time.  This is scalar only; no content remapping.
        self.intent_gain = nn.Parameter(
            torch.tensor(1.0)
        )

    @classmethod
    def grow_from_admitted(cls, admitted):
        return cls(admitted)

    def load_frozen_parent_executive(self, state_dict):
        self.frozen_parent_executive.load_state_dict(
            state_dict,
            strict=True,
        )
        for p in self.frozen_parent_executive.parameters():
            p.requires_grad_(False)

    def initial_state(self, batch_size, device):
        return self.admitted.initial_state(
            batch_size,
            device,
        )

    def initial_memory(self, batch_size, device):
        return self.admitted.initial_memory(
            batch_size,
            device,
        )

    def initial_option_state(self, batch_size, device):
        return self.progress.initial_state(
            batch_size,
            device,
        )

    def components(
        self,
        image,
        context,
        memory,
        prev_action,
        hidden,
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
        base_logits = self.base.motor(
            admitted_workspace
        )
        return (
            hist,
            recall_residual,
            recall_aux,
            admitted_workspace,
            base_logits,
        )

    def advance_progress(
        self,
        protected_hidden,
        historical_workspace,
        base_logits,
        option_state,
        rule_id,
    ):
        active = (
            rule_id.ne(RULE_NONE)
            .to(protected_hidden.dtype)
        )
        return self.progress(
            protected_hidden,
            historical_workspace,
            base_logits,
            option_state,
            active,
        )

    def raw_intent(
        self,
        canonical_recall,
        rule_id,
    ):
        raw, aux = (
            self.frozen_parent_executive.compute_intent(
                canonical_recall,
                rule_id,
            )
        )
        return raw, aux

    def apply_intent(
        self,
        canonical_recall,
        rule_id,
        base_logits,
        execution_gate,
    ):
        raw_intent, intent_aux = self.raw_intent(
            canonical_recall,
            rule_id,
        )
        delta = (
            raw_intent
            * execution_gate
            * self.intent_gain
        )
        logits = base_logits + delta
        return logits, delta, {
            **intent_aux,
            "raw_intent": raw_intent,
        }

    def step(
        self,
        image,
        context,
        memory,
        prev_action,
        hidden,
        option_state,
        rule_id,
    ):
        (
            hist,
            recall_residual,
            recall_aux,
            admitted_workspace,
            base_logits,
        ) = self.components(
            image,
            context,
            memory,
            prev_action,
            hidden,
        )

        (
            execution_gate,
            gate_logit,
            option_state_next,
            progress_aux,
        ) = self.advance_progress(
            hist["hidden"],
            hist["historical_workspace"],
            base_logits,
            option_state,
            rule_id,
        )

        logits, delta, intent_aux = self.apply_intent(
            recall_residual,
            rule_id,
            base_logits,
            execution_gate,
        )

        next_memory, mem_aux = self.admitted.memory(
            hist["hidden"],
            memory,
        )

        return {
            **hist,
            "workspace_for_world":
                admitted_workspace,
            "admitted_workspace":
                admitted_workspace,
            "logits": logits,
            "base_logits": base_logits,
            "executive_intent_delta": delta,
            "raw_intent": intent_aux["raw_intent"],
            "execution_gate": execution_gate,
            "execution_gate_logit": gate_logit,
            "option_state": option_state,
            "option_state_next":
                option_state_next,
            "memory_in": memory,
            "memory_next": next_memory,
            "memory_candidate":
                mem_aux["candidate"],
            "memory_write_gate":
                mem_aux["write_gate"],
            "recall_residual":
                recall_residual,
            "recall_decoded":
                recall_aux["decoded"],
            "progress_hidden_features":
                progress_aux[
                    "protected_hidden_features"
                ],
            "progress_workspace_features":
                progress_aux[
                    "workspace_features"
                ],
            "progress_action_features":
                progress_aux[
                    "action_features"
                ],
        }

    def trainable_parameters(self):
        return (
            list(self.progress.parameters())
            + [self.intent_gain]
        )
