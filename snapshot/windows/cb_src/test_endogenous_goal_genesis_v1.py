#!/usr/bin/env python3
from continuous_brain.endogenous_goal_genesis_v1 import (
    PRIMARY_A,
    PRIMARY_B,
    EndogenousGoalGenerator,
    GoalLifecycle,
)


def main():
    g = EndogenousGoalGenerator()

    a = g.generate(
        [0.10, 0.90],
        generation=0,
        cycle=0,
    )
    assert a is not None
    assert a.primary == PRIMARY_A
    assert abs(a.target_reliability - .22) < 1e-9

    b = g.generate(
        [0.90, 0.10],
        generation=0,
        cycle=0,
    )
    assert b is not None
    assert b.primary == PRIMARY_B

    neutral = g.generate(
        [0.5, 0.5],
        generation=0,
        cycle=0,
    )
    assert neutral is None

    life = GoalLifecycle()
    life.begin(a)
    assert life.active_goal == a
    life.complete_active(3)
    assert life.active_goal is None
    assert len(life.completed_goals) == 1
    assert life.generation == 1

    print("ENDOGENOUS_GOAL_GENESIS_SMOKE: PASS")


if __name__ == "__main__":
    main()
