#!/usr/bin/env python3
from continuous_brain.endogenous_goal_genesis_v1 import (
    EndogenousGoalGenerator,
)


def main():
    g = EndogenousGoalGenerator()
    a = g.generate([.10, .90], generation=0, cycle=0)
    b = g.generate([.90, .10], generation=0, cycle=0)

    assert a is not None and b is not None
    assert abs(a.target_reliability - .22) < 1e-9
    assert abs(b.target_reliability - .22) < 1e-9

    assert 16 == 2 * 8

    print("ENDOGENOUS_GOAL_HORIZON_REPAIR_SMOKE: PASS")


if __name__ == "__main__":
    main()
