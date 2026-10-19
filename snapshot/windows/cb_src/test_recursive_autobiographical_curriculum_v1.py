#!/usr/bin/env python3
from continuous_brain.endogenous_goal_genesis_v1 import (
    EndogenousGoalGenerator,
)
from continuous_brain.recursive_autobiographical_curriculum_v1 import (
    goal_is_advanced,
    same_goal_semantics,
)


def main():
    g = EndogenousGoalGenerator()

    first = g.generate(
        [.10, .90],
        generation=0,
        cycle=0,
    )
    same = g.generate(
        [.10, .90],
        generation=1,
        cycle=8,
    )
    advanced = g.generate(
        [.25, .90],
        generation=1,
        cycle=8,
    )
    switched = g.generate(
        [.90, .20],
        generation=1,
        cycle=8,
    )

    assert first is not None
    assert same_goal_semantics(first, same)
    assert not goal_is_advanced(first, same)
    assert goal_is_advanced(first, advanced)
    assert goal_is_advanced(first, switched)

    print("RECURSIVE_AUTOBIOGRAPHICAL_CURRICULUM_SMOKE: PASS")


if __name__ == "__main__":
    main()
