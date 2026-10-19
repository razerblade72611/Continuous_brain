#!/usr/bin/env python3
import torch

from continuous_brain.persistent_self_autobiographical_v1 import (
    GOAL_COMMITMENT_A,
    PersistentSelfState,
)
from continuous_brain.joint_self_world_prediction_v1 import (
    ACTION_DIM,
    JointSelfWorldModel,
    choose_action,
    predicted_self_vector,
    self_prediction_error,
)


def main():
    torch.manual_seed(30001)

    model = JointSelfWorldModel(
        hidden_dim=64,
        latent_dim=48,
    )

    state = PersistentSelfState.create(30001)
    state.unresolved_a = .9
    state.unresolved_b = .1
    state.confidence = .6
    state.uncertainty = .2
    state.prediction_error_ema = .1
    state.agency_ema = .5
    state.active_goal = GOAL_COMMITMENT_A

    world = torch.tensor(
        [[.3, .2, .5, .3]],
        dtype=torch.float32,
    )
    self_v = state.vector().unsqueeze(0)
    action = torch.tensor([1], dtype=torch.long)

    out = model(world, self_v, action)

    assert tuple(out["next_world"].shape) == (1, 4)
    assert tuple(out["next_self_continuous"].shape) == (1, 6)
    assert tuple(out["next_goal_logits"].shape) == (1, 4)
    assert tuple(out["current_self_reconstruction"].shape) == (1, 10)

    pred = predicted_self_vector(out)
    assert tuple(pred.shape) == (1, 10)

    err = self_prediction_error(pred[0], self_v[0])
    assert err >= 0.0

    choice = choose_action(
        model,
        world[0],
        self_v[0],
        self_weight=1.0,
    )
    assert 0 <= choice["action"] < ACTION_DIM
    assert len(choice["scores"]) == ACTION_DIM

    print("JOINT_SELF_WORLD_PREDICTION_SMOKE: PASS")


if __name__ == "__main__":
    main()
