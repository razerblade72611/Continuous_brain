#!/usr/bin/env python3
import torch
from torch import nn

from continuous_brain.persistent_self_autobiographical_v1 import (
    GOAL_COMMITMENT_A,
    PersistentSelfState,
)
from continuous_brain.self_broadcast_centrality_v1 import (
    SelfBroadcastHub,
    apply_workspace_broadcast,
    self_input_from_state,
)

class FakeBase:
    def __init__(self, workspace_dim=16, action_dim=5):
        self.motor = nn.Linear(workspace_dim, action_dim, bias=False)

def main():
    torch.manual_seed(29)
    state = PersistentSelfState.create(29001)
    state.unresolved_a = 1.0
    state.unresolved_b = .1
    state.active_goal = GOAL_COMMITMENT_A

    x = self_input_from_state(state).unsqueeze(0)
    assert tuple(x.shape) == (1, 42)

    hub = SelfBroadcastHub(
        workspace_dim=16,
        hidden_dim=32,
        broadcast_dim=24,
    )
    base = FakeBase(16, 5)
    workspace = torch.randn(1, 16)

    on = hub(x, recruitment=1.0, enabled=True)
    assert tuple(on["broadcast"].shape) == (1, 24)
    assert tuple(on["workspace_residual"].shape) == (1, 16)
    assert tuple(on["retrieval_logits"].shape) == (1, 3)
    assert tuple(on["goal_logits"].shape) == (1, 4)
    assert tuple(on["planning_mode_logits"].shape) == (1, 2)
    assert tuple(on["identity_reconstruction"].shape) == (1, 32)

    off = hub(x, recruitment=1.0, enabled=False)
    assert torch.count_nonzero(off["broadcast"]).item() == 0
    assert torch.count_nonzero(off["workspace_residual"]).item() == 0
    assert torch.count_nonzero(off["retrieval_logits"]).item() == 0
    assert torch.count_nonzero(off["goal_logits"]).item() == 0
    assert torch.count_nonzero(off["planning_mode_logits"]).item() == 0

    combined, off_logits = apply_workspace_broadcast(
        base, workspace, off
    )
    base_logits = base.motor(workspace)
    assert torch.equal(combined, workspace)
    assert torch.equal(off_logits, base_logits)

    unrecruited = hub(x, recruitment=0.0, enabled=True)
    assert torch.count_nonzero(unrecruited["broadcast"]).item() == 0

    print("SELF_BROADCAST_CENTRALITY_SMOKE: PASS")

if __name__ == "__main__":
    main()
