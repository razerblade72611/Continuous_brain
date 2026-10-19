"""Function-preserving competitive broadcast growth for :class:`InfantBrain`.

This module is deliberately additive.  The complete learned InfantBrain stays
frozen.  A zero-initialized competitive pathway can grow around it and can be
turned off at any time to recover the original checkpoint behavior exactly.

The competition acts on the four streams already entering ``CorticalNucleus``:

    fused vision, projected context, projected memory, previous-action embedding

A shared condition uses one selector for three real, differentiable consumers:
recurrent state, motor policy, and world model.  The independent control uses a
separate selector for each consumer.  We intentionally do *not* repurpose the
``values`` head: in this project it predicts the six-dimensional care vector,
not a generic RL value, and CuedDetourWorld has no honest care labels.
"""
from __future__ import annotations

from typing import Mapping

import torch
from torch import nn

from .infant import InfantBrain, InfantConfig


class StreamCompetition(nn.Module):
    """Top-down selection over the four nucleus input streams.

    ``hidden_prev`` is the query.  The query layer is zero-initialized, so the
    selector starts at exactly uniform weights.  Function preservation comes
    from the zero-initialized consumer residuals, not from uniform selection.
    """

    def __init__(self, nucleus_dim: int, bus_dim: int):
        super().__init__()
        self.bus_dim = int(bus_dim)
        self.query = nn.Linear(int(nucleus_dim), int(bus_dim))
        nn.init.zeros_(self.query.weight)
        nn.init.zeros_(self.query.bias)

    def forward(self, hidden_prev: torch.Tensor, streams: list[torch.Tensor]) -> dict:
        if len(streams) != 4:
            raise ValueError(f"expected four workspace streams, got {len(streams)}")
        q = self.query(hidden_prev)
        stacked = torch.stack(streams, dim=1)
        scores = torch.einsum("bw,bsw->bs", q, stacked) / (self.bus_dim ** 0.5)
        weights = torch.softmax(scores, dim=-1)
        broadcast = torch.einsum("bs,bsw->bw", weights, stacked)
        return {"weights": weights, "broadcast": broadcast}


class IndependentSelectionTriple(nn.Module):
    """Capacity-richer control with independent selectors per real consumer."""

    def __init__(self, nucleus_dim: int, bus_dim: int):
        super().__init__()
        self.recurrent_gate = StreamCompetition(nucleus_dim, bus_dim)
        self.motor_gate = StreamCompetition(nucleus_dim, bus_dim)
        self.world_gate = StreamCompetition(nucleus_dim, bus_dim)

    def forward(self, hidden_prev: torch.Tensor, streams: list[torch.Tensor]) -> dict:
        r = self.recurrent_gate(hidden_prev, streams)
        m = self.motor_gate(hidden_prev, streams)
        w = self.world_gate(hidden_prev, streams)
        return {
            "recurrent_broadcast": r["broadcast"],
            "motor_broadcast": m["broadcast"],
            "world_broadcast": w["broadcast"],
            "recurrent_weights": r["weights"],
            "motor_weights": m["weights"],
            "world_weights": w["weights"],
        }


class ZeroInitResidual(nn.Module):
    """A zero-initialized additive adapter."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(int(in_features), int(out_features))
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, broadcast: torch.Tensor) -> torch.Tensor:
        return self.linear(broadcast)


def _parent_dims(model: InfantBrain) -> dict[str, int]:
    return {name: int(head.out_features) for name, head in model.vision_parent_heads.items()}


class CompetitiveInfantBrain(nn.Module):
    """Frozen InfantBrain plus an ablatable competitive broadcast pathway.

    When competition is enabled after training, the selected broadcast can:
      1. add a residual to recurrent hidden state, making selection load-bearing
         for retention across future steps;
      2. add a residual to the motor workspace;
      3. add a residual to the world-model workspace.

    At construction all three residuals are exactly zero.  Disabled mode always
    uses the untouched frozen path, which makes checkpoint preservation directly
    testable instead of relying on approximate regularization.
    """

    def __init__(
        self,
        cfg: InfantConfig | None = None,
        *,
        use_shared: bool = True,
        vision_parent_dims: Mapping[str, int] | None = None,
    ):
        super().__init__()
        self.cfg = cfg or InfantConfig()
        nd, bd = int(self.cfg.nucleus_dim), int(self.cfg.bus_dim)
        self.base = InfantBrain(self.cfg, vision_parent_dims=vision_parent_dims)
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.base.eval()

        self.use_shared = bool(use_shared)
        self.competition = (
            StreamCompetition(nd, bd)
            if self.use_shared
            else IndependentSelectionTriple(nd, bd)
        )
        self.recurrent_residual = ZeroInitResidual(bd, nd)
        self.motor_residual = ZeroInitResidual(bd, bd)
        self.world_residual = ZeroInitResidual(bd, bd)

    def train(self, mode: bool = True):
        # ``super().train`` is needed for the trainable pathway.  The frozen base
        # stays in eval mode so BatchNorm running buffers cannot silently drift.
        super().train(mode)
        self.base.eval()
        return self

    @classmethod
    def grow_from(cls, old: InfantBrain, *, use_shared: bool = True) -> "CompetitiveInfantBrain":
        grown = cls(old.cfg, use_shared=use_shared, vision_parent_dims=_parent_dims(old))
        grown.base.load_state_dict(old.state_dict(), strict=True)
        for p in grown.base.parameters():
            p.requires_grad_(False)
        grown.base.eval()
        return grown

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def initial_state(self, n: int, device) -> torch.Tensor:
        return self.base.nucleus.initial_state(n, device)

    def _workspace_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        # Exactly the same frozen projection used by CorticalNucleus.forward.
        return self.base.nucleus.growth(self.base.nucleus.workspace(hidden))

    def step(
        self,
        images: torch.Tensor,
        context: torch.Tensor,
        memory: torch.Tensor,
        prev_action: torch.Tensor,
        hidden: torch.Tensor,
        *,
        use_competition: bool = True,
    ) -> dict:
        semantic = self.base.vision(images)
        spatial = self.base.spatial(images)
        fused = self.base.fuse_vision(semantic, spatial)
        nuc = self.base.nucleus(fused, context, memory, prev_action, hidden)
        base_hidden = nuc["hidden"]
        base_workspace = nuc["workspace"]

        if not use_competition:
            return {
                "hidden": base_hidden,
                "logits": self.base.motor(base_workspace),
                "value": self.base.values(base_workspace),
                "workspace_for_world": base_workspace,
                "router": nuc["router"],
                "competition": None,
            }

        # Reuse the frozen projections already defined by CorticalNucleus.  The
        # competitive branch therefore observes the same four streams rather than
        # introducing a second, independently learned encoder stack.
        streams = [
            fused,
            self.base.nucleus.context(context),
            self.base.nucleus.memory(memory),
            self.base.nucleus.action(prev_action),
        ]

        if self.use_shared:
            comp = self.competition(hidden, streams)
            recurrent_b = motor_b = world_b = comp["broadcast"]
        else:
            comp = self.competition(hidden, streams)
            recurrent_b = comp["recurrent_broadcast"]
            motor_b = comp["motor_broadcast"]
            world_b = comp["world_broadcast"]

        competitive_hidden = base_hidden + self.recurrent_residual(recurrent_b)
        recurrent_workspace = self._workspace_from_hidden(competitive_hidden)
        motor_workspace = recurrent_workspace + self.motor_residual(motor_b)
        world_workspace = recurrent_workspace + self.world_residual(world_b)

        return {
            "hidden": competitive_hidden,
            "logits": self.base.motor(motor_workspace),
            # Care prediction stays on the frozen base workspace; this experiment
            # has no honest care target and therefore does not train that consumer.
            "value": self.base.values(base_workspace),
            "workspace_for_world": world_workspace,
            "router": nuc["router"],
            "competition": comp,
        }

    def predict_world(
        self,
        workspace_for_world: torch.Tensor,
        action_onehot: torch.Tensor,
        **kwargs,
    ) -> dict:
        return self.base.world(workspace_for_world, action_onehot, **kwargs)
