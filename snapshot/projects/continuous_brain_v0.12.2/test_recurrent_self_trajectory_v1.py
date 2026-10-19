#!/usr/bin/env python3
import torch

from continuous_brain.recurrent_self_trajectory_v1 import (
    MODE_INVESTIGATE,
    MODE_NORMAL,
    PRIMARY_A,
    RecurrentSelfTrajectoryModel,
    active_goal_from_intention,
    rollout,
    update_explicit_intention,
)


def main():
    torch.manual_seed(31001)
    model=RecurrentSelfTrajectoryModel(
        hidden_dim=48,
        context_dim=32,
    )

    world=torch.tensor([.3,.2,.5,.3],dtype=torch.float32)
    selfc=torch.tensor([.8,.1,.6,.2,.1,.5],dtype=torch.float32)

    r=rollout(
        model,
        world,
        selfc,
        PRIMARY_A,
        MODE_NORMAL,
        [1,1,0],
    )
    assert tuple(r["worlds"].shape)==(3,4)
    assert tuple(r["selfs"].shape)==(3,6)
    assert len(r["primaries"])==3
    assert len(r["modes"])==3

    # Explicit primary survives temporary investigate.
    high_surprise=selfc.clone()
    high_surprise[3]=.90
    high_surprise[4]=.70
    x=update_explicit_intention(
        PRIMARY_A,
        MODE_NORMAL,
        high_surprise,
    )
    assert x.primary==PRIMARY_A
    assert x.mode==MODE_INVESTIGATE

    resolved=high_surprise.clone()
    resolved[3]=.30
    resolved[4]=.20
    y=update_explicit_intention(
        x.primary,
        x.mode,
        resolved,
    )
    assert y.primary==PRIMARY_A
    assert y.mode==MODE_NORMAL

    assert active_goal_from_intention(
        y.primary,y.mode
    )==1

    print("RECURRENT_SELF_TRAJECTORY_SMOKE: PASS")


if __name__=="__main__":
    main()
