#!/usr/bin/env python3
from continuous_brain.endogenous_goal_genesis_v1 import (
    EndogenousGoalGenerator,
)


def main():
    g = EndogenousGoalGenerator()

    assert g.generate([.10, .90], generation=0, cycle=0) is not None
    assert g.generate([.90, .10], generation=0, cycle=0) is not None
    assert g.generate([.50, .50], generation=0, cycle=0) is None

    ids = list(range(100000, 100128))
    assert ids[0] == 100000
    assert ids[-1] == 100127
    assert len(ids) == 128
    assert 16 > 8

    print("ENDOGENOUS_GOAL_GENESIS_TEST_SMOKE: PASS")


if __name__ == "__main__":
    main()
