"""Shared-selection competitive gating for DevelopmentBrain, plus an
independent-selection control -- redesigned this round as frozen base
plus trainable additive residuals (the standard adapter/LoRA-style
pattern: a frozen pretrained computation, plus a small trainable branch
summed on top), specifically because the previous "grow the weight
matrix and zero the new columns" design could not make a precise,
by-construction claim about which weights stay frozen through training.
This one can.

NOT executed against real PyTorch in the sandbox that wrote this (no
network access to install torch there). Every zero-init and gradient-
existence claim below has a NumPy-only verification in verify_zero_init.py,
actually run in this sandbox -- that file's output is real; this file's
correctness under real torch autograd is reviewed, not executed, and
should be checked in the environment that has PyTorch before being
trusted past the reasoning given here.
"""
from __future__ import annotations

import torch
from torch import nn

from .model import DevelopmentBrain, BrainConfig


class StreamCompetition(nn.Module):
    """hidden_{t-1} is the query; the three input-stream embeddings are
    the candidates. Zero-initialized query -> uniform 1/3 weights at
    construction (verified in verify_zero_init.py). Uniform weights are
    NOT a no-op by themselves; the no-op property for the whole system
    comes from the consumer-side residual layers below being zero-init
    and summed additively onto a FROZEN base output, not from this
    module's weights being zero."""

    def __init__(self, hidden_dim: int, stream_width: int):
        super().__init__()
        self.stream_width = stream_width
        self.query = nn.Linear(hidden_dim, stream_width)
        nn.init.zeros_(self.query.weight)
        nn.init.zeros_(self.query.bias)

    def forward(self, hidden_prev: torch.Tensor, streams: list[torch.Tensor]) -> dict:
        q = self.query(hidden_prev)
        stacked = torch.stack(streams, dim=1)
        scores = torch.einsum("bw,bsw->bs", q, stacked) / (self.stream_width ** 0.5)
        weights = torch.softmax(scores, dim=-1)
        broadcast = torch.einsum("bs,bsw->bw", weights, stacked)
        return {"weights": weights, "broadcast": broadcast}


class IndependentSelection(nn.Module):
    """Control: two separately-parameterized StreamCompetition instances.
    workspace_gate's output is consumed by the (differentiable) workspace
    and heads, exactly like the shared case. memory_gate's output is
    consumed ONLY by ExperienceMemory.add() -- a plain symbolic buffer
    write, not a differentiable operation. That means memory_gate NEVER
    receives gradient, from anything, ever: not a bug, a real structural
    property of what "independent selection" means once memory-write
    isn't part of the loss's computation graph. Reported explicitly in
    PROPOSAL.md's parameter count, not hidden."""

    def __init__(self, hidden_dim: int, stream_width: int):
        super().__init__()
        self.workspace_gate = StreamCompetition(hidden_dim, stream_width)
        self.memory_gate = StreamCompetition(hidden_dim, stream_width)

    def forward(self, hidden_prev: torch.Tensor, streams: list[torch.Tensor]) -> dict:
        ws = self.workspace_gate(hidden_prev, streams)
        with torch.no_grad():  # explicit, not incidental: memory_gate structurally cannot train, see class docstring
            mem = self.memory_gate(hidden_prev, streams)
        return {"weights": ws["weights"], "broadcast": ws["broadcast"],
                "memory_weights": mem["weights"], "memory_broadcast": mem["broadcast"]}


class ZeroInitResidual(nn.Module):
    """The consumer-side half of the adapter pattern: takes ONLY the new
    (broadcast) input and produces an additive correction, zero at
    construction. `frozen_output + self(broadcast)` starts exactly equal
    to `frozen_output` and can only move away from it as THIS module's
    own (separate, small) parameters train -- the frozen base layer is
    never touched."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, broadcast: torch.Tensor) -> torch.Tensor:
        return self.linear(broadcast)


class CompetitiveDevelopmentBrain(nn.Module):
    """DevelopmentBrain (self.base), frozen entirely, plus StreamCompetition
    or IndependentSelection and four ZeroInitResidual branches (workspace,
    policy, q, world). Only the competition module and the four residuals
    are trainable; every base.* parameter has requires_grad=False, set
    once in __init__ and never toggled, so "which weights stay frozen" is
    answerable by reading one line, not by re-deriving it from training
    behavior after the fact."""

    def __init__(self, cfg: BrainConfig | None = None, use_shared: bool = True):
        super().__init__()
        self.cfg = cfg or BrainConfig()
        d, h = self.cfg.width, self.cfg.hidden
        self.base = DevelopmentBrain(self.cfg)
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.competition = (StreamCompetition if use_shared else IndependentSelection)(h, d)
        self.use_shared = use_shared
        # Residual correction applied to each consumer's OUTPUT, not fused
        # into the frozen GRU's internal gate math -- nn.GRUCell's public
        # forward() doesn't expose per-gate internals to fuse into without
        # reimplementing the cell, and doing that quietly would be exactly
        # the kind of unreviewed complexity this round asked to avoid.
        # Stated plainly: broadcast's influence on the recurrent state is
        # "added to this step's hidden output," not "fused into this
        # step's gate computation." A real design simplification, not
        # hidden as something stronger.
        self.workspace_residual = ZeroInitResidual(d, h)
        self.policy_residual = ZeroInitResidual(d, d)
        self.q_residual = ZeroInitResidual(d, d)
        self.world_residual = ZeroInitResidual(d, d)

    @classmethod
    def grow_from(cls, old: DevelopmentBrain, use_shared: bool = True) -> "CompetitiveDevelopmentBrain":
        grown = cls(old.cfg, use_shared=use_shared)
        grown.base.load_state_dict(old.state_dict())
        for p in grown.base.parameters():
            p.requires_grad_(False)
        return grown

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def initial(self, n, device):
        return torch.zeros(n, self.cfg.hidden, device=device)

    def forward(self, grid, sensors, memory, style, hidden, *, use_competition: bool = True):
        perception = self.base.perception(grid)
        interoception = self.base.interoception(torch.cat((sensors, style), -1))
        memory_emb = self.base.memory_adapter(memory)
        x = torch.cat((perception, interoception, memory_emb), -1)  # unchanged from base -- frozen path
        base_hidden = self.base.workspace(x, hidden)                # frozen GRU, always computed, always identical
                                                                      # to plain DevelopmentBrain given the same inputs

        if not use_competition:
            new_hidden = base_hidden
            broadcast = torch.zeros(grid.shape[0], self.cfg.width, device=grid.device)
            logits, q = self.base.policy(new_hidden), self.base.q(new_hidden)
            memory_write_content = None
        else:
            comp = self.competition(hidden, [perception, interoception, memory_emb])
            broadcast = comp["broadcast"]
            memory_write_content = broadcast if self.use_shared else comp["memory_broadcast"]
            new_hidden = base_hidden + self.workspace_residual(broadcast)   # zero at init -> new_hidden == base_hidden
            logits = self.base.policy(new_hidden) + self.policy_residual(broadcast)
            q = self.base.q(new_hidden) + self.q_residual(broadcast)

        result = {"hidden": new_hidden, "logits": logits, "q": q, "broadcast": broadcast}
        if use_competition:
            result["competition_weights"] = comp["weights"]
            result["memory_write_content"] = memory_write_content
        return result

    def predict(self, hidden, broadcast, action_onehot):
        base_pred = self.base.world(torch.cat((hidden, action_onehot), -1))
        pred = base_pred + self.world_residual(broadcast)
        return {"body": torch.sigmoid(pred[..., :3]), "reward": pred[..., 3],
                "collision_logit": pred[..., 4], "resource_logits": pred[..., 5:8]}

    def choose(self, result):
        return result["logits"] + .10 * torch.tanh(result["q"])
