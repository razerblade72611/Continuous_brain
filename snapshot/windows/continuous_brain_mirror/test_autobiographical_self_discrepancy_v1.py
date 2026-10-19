#!/usr/bin/env python3
import torch

from continuous_brain.autobiographical_self_discrepancy_v1 import (
    HISTORY_LEN,
    RECORD_FEATURE_DIM,
    AutobiographicalSelfModel,
    AutobiographicalTrajectoryMemory,
    TrajectoryMemoryRecord,
)


def make_record(primary):
    expected_self=torch.zeros(4,6)
    actual_self=expected_self.clone()
    actual_self[:,3]=.2
    expected_world=torch.zeros(4,4)
    actual_world=expected_world.clone()
    expected_modes=torch.zeros(4,dtype=torch.long)
    actual_modes=torch.ones(4,dtype=torch.long)

    return TrajectoryMemoryRecord(
        episode_id=1,
        primary=primary,
        expected_self=expected_self,
        actual_self=actual_self,
        expected_world=expected_world,
        actual_world=actual_world,
        expected_modes=expected_modes,
        actual_modes=actual_modes,
    )


def main():
    a=make_record(1)
    f=a.feature()
    assert tuple(f.shape)==(RECORD_FEATURE_DIM,)

    memory=AutobiographicalTrajectoryMemory([a,make_record(2)])
    x,m=memory.feature_tensor(HISTORY_LEN)
    assert tuple(x.shape)==(HISTORY_LEN,RECORD_FEATURE_DIM)
    assert tuple(m.shape)==(HISTORY_LEN,)
    assert int(m.sum().item())==2

    ablated=memory.discrepancy_ablation()
    xa,ma=ablated.feature_tensor(HISTORY_LEN)
    # Primary one-hot survives; all discrepancy-derived fields vanish.
    assert torch.count_nonzero(xa[:,:2]).item()==2
    assert torch.count_nonzero(xa[:,2:]).item()==0

    model=AutobiographicalSelfModel()
    out=model(x.unsqueeze(0),m.unsqueeze(0))
    assert tuple(out["reliability"].shape)==(1,2)

    # Empty autobiography is exact neutral by architecture.
    z=torch.zeros(1,HISTORY_LEN,RECORD_FEATURE_DIM)
    zm=torch.zeros(1,HISTORY_LEN,dtype=torch.bool)
    zo=model(z,zm)
    assert torch.equal(
        zo["reliability"][0],
        torch.tensor([.5,.5]),
    )

    print("AUTOBIOGRAPHICAL_SELF_DISCREPANCY_SMOKE: PASS")


if __name__=="__main__":
    main()
