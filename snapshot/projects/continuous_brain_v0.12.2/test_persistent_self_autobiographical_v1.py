#!/usr/bin/env python3
from pathlib import Path
import tempfile

import torch

from continuous_brain.persistent_self_autobiographical_v1 import (
    ACTION_WAIT,
    GOAL_COMMITMENT_A,
    GOAL_COMMITMENT_B,
    GOAL_INVESTIGATE,
    GOAL_NONE,
    AgencySelfModel,
    PersistentSelfCore,
    PersistentSelfState,
    load_self_bundle,
    save_self_bundle,
)


def main():
    model = AgencySelfModel(hidden_dim=32)
    model.eval()

    state = PersistentSelfState.create(1234)
    assert state.identity_embedding.shape == (32,)
    assert abs(float(state.identity_embedding.norm()) - 1.0) < 1e-5

    core = PersistentSelfCore(model, state=state)
    core.add_commitment(GOAL_COMMITMENT_A, 1.0)
    core.state.unresolved_b = .25
    assert core.choose_goal() == GOAL_COMMITMENT_A

    completion = core.record_completion(GOAL_COMMITMENT_A)
    assert completion.completed_goal == GOAL_COMMITMENT_A
    assert core.choose_goal() == GOAL_COMMITMENT_B

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "self.pt"
        save_self_bundle(path, core, {"smoke": True})
        loaded, payload = load_self_bundle(path, model)
        assert loaded.state.identity_token == core.state.identity_token
        assert torch.equal(
            loaded.state.identity_embedding,
            core.state.identity_embedding,
        )
        assert loaded.memory.to_list() == core.memory.to_list()

    # Endogenous information-seeking is generated from internal surprise.
    s = PersistentSelfState.create(555)
    s.prediction_error_ema = .80
    s.uncertainty = .30
    c = PersistentSelfCore(model, state=s)
    assert c.choose_goal() == GOAL_INVESTIGATE

    s2 = s.clone()
    s2.prediction_error_ema = 0.0
    c2 = PersistentSelfCore(model, state=s2)
    assert c2.choose_goal() == GOAL_NONE

    # Attribution intervention must alter world-attributed error.
    b = PersistentSelfState.create(777)
    b.prediction_error_ema = .20
    b.uncertainty = .30

    self_attr = PersistentSelfCore(model, state=b)
    world_attr = PersistentSelfCore(model, state=b)

    self_attr.observe_event(
        action=ACTION_WAIT,
        intended_change=(0.0, 0.0),
        observed_change=(1.0, 0.0),
        action_success=1.0,
        prediction_error=1.0,
        forced_agency_probability=1.0,
    )
    world_attr.observe_event(
        action=ACTION_WAIT,
        intended_change=(0.0, 0.0),
        observed_change=(1.0, 0.0),
        action_success=1.0,
        prediction_error=1.0,
        forced_agency_probability=0.0,
    )
    assert (
        world_attr.state.prediction_error_ema
        > self_attr.state.prediction_error_ema
    )
    assert world_attr.state.uncertainty > self_attr.state.uncertainty

    print("PERSISTENT_SELF_AUTOBIOGRAPHICAL_SMOKE: PASS")


if __name__ == "__main__":
    main()
