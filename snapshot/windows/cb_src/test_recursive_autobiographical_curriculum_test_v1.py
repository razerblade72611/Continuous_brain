#!/usr/bin/env python3

from continuous_brain.recursive_autobiographical_curriculum_v1 import (
    MIN_SAME_PRIMARY_TARGET_ADVANCE,
)


def main():
    ids = list(range(120000, 120128))
    assert ids[0] == 120000
    assert ids[-1] == 120127
    assert len(ids) == 128

    assert abs(
        MIN_SAME_PRIMARY_TARGET_ADVANCE - .05
    ) < 1e-12

    print(
        "RECURSIVE_AUTOBIOGRAPHICAL_CURRICULUM_TEST_SMOKE: PASS"
    )


if __name__ == "__main__":
    main()
