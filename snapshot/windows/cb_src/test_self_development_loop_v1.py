#!/usr/bin/env python3
import torch

from continuous_brain.autobiographical_self_discrepancy_v1 import (
    AutobiographicalTrajectoryMemory,
)
from continuous_brain.self_development_loop_v1 import (
    STRATEGY_DIRECT,
    STRATEGY_SCAFFOLD,
    DevelopmentalController,
    DevelopmentalState,
    capability_update,
    strategy_failure_probability,
)


def main():
    controller = DevelopmentalController(
        direct_threshold=.50
    )
    assert controller.choose(.20) == STRATEGY_SCAFFOLD
    assert controller.choose(.50) == STRATEGY_DIRECT
    assert controller.choose(.90) == STRATEGY_DIRECT

    low = .20
    assert (
        strategy_failure_probability(
            low,
            STRATEGY_SCAFFOLD,
        )
        <
        strategy_failure_probability(
            low,
            STRATEGY_DIRECT,
        )
    )

    s = DevelopmentalState(
        capability_a=.20,
        capability_b=.90,
        cycle=0,
    )
    before = s.capability_for(1)
    after = capability_update(
        before,
        success=True,
        strategy=STRATEGY_SCAFFOLD,
    )
    assert after > before
    s.set_capability_for(1, after)
    assert s.capability_for(1) == after

    # Empty memory object is valid; the frozen v0.32 model supplies exact
    # [0.5,0.5] neutrality when it is evaluated in the full assay.
    memory = AutobiographicalTrajectoryMemory()
    x, m = memory.feature_tensor()
    assert x.ndim == 2
    assert m.sum().item() == 0

    print("SELF_DEVELOPMENT_LOOP_SMOKE: PASS")


if __name__ == "__main__":
    main()
