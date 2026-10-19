#!/usr/bin/env python3
import numpy as np

from continuous_brain.varied_cued_detour_world import (
    DEV_CONDITION_RANGE,
    FAMILY_SIZE,
    LEFT,
    RIGHT,
    TEST_CONDITION_RANGE,
    TRAIN_CONDITION_RANGE,
    VariedCuedDetourWorld,
    generate_task_spec,
)


def test_family_size_and_splits():
    assert FAMILY_SIZE == 2360
    assert len(TRAIN_CONDITION_RANGE) == 1600
    assert len(DEV_CONDITION_RANGE) == 380
    assert len(TEST_CONDITION_RANGE) == 380
    assert set(TRAIN_CONDITION_RANGE).isdisjoint(DEV_CONDITION_RANGE)
    assert set(TRAIN_CONDITION_RANGE).isdisjoint(TEST_CONDITION_RANGE)
    assert set(DEV_CONDITION_RANGE).isdisjoint(TEST_CONDITION_RANGE)


def test_condition_ids_do_not_wrap():
    generate_task_spec(0)
    generate_task_spec(FAMILY_SIZE - 1)
    try:
        generate_task_spec(FAMILY_SIZE)
    except ValueError:
        pass
    else:
        raise AssertionError("out-of-range condition_id silently wrapped")


def test_first_256_structural_specs_are_unique():
    sigs = {
        generate_task_spec(i).structural_signature()
        for i in range(256)
    }
    assert len(sigs) == 256


def test_pair_becomes_sensor_identical_after_cue_hides():
    for condition_id in (0, 1, 17, 63, 127, 255):
        left = VariedCuedDetourWorld(condition_id, cue="left")
        right = VariedCuedDetourWorld(condition_id, cue="right")

        assert left.spec == right.spec
        assert not np.array_equal(left._obs(), right._obs())
        assert not np.array_equal(
            left.render_rgb(96), right.render_rgb(96)
        )

        hidden_identical = 0

        while True:
            if left.steps >= left.cue_steps:
                assert np.array_equal(left._obs(), right._obs())
                assert np.array_equal(
                    left.render_rgb(96), right.render_rgb(96)
                )
                hidden_identical += 1

            a_l = left.teacher_action()
            a_r = right.teacher_action()

            if a_l in (LEFT, RIGHT) or a_r in (LEFT, RIGHT):
                assert a_l == LEFT
                assert a_r == RIGHT
                break

            assert a_l == a_r
            _, _, d_l, _ = left.step(a_l)
            _, _, d_r, _ = right.step(a_r)
            assert not d_l
            assert not d_r

        assert hidden_identical >= 3


def test_teacher_solves_both_counterfactual_members():
    for condition_id in (0, 3, 11, 47, 101, 255):
        totals = []
        for cue in ("left", "right"):
            env = VariedCuedDetourWorld(condition_id, cue=cue)
            info = {"reached_correct": False}
            safety = 0
            while not env.done:
                _, _, _, info = env.step(env.teacher_action())
                safety += 1
                assert safety < 120
            assert info["reached_correct"] is True
            totals.append(env.steps)
        assert totals[0] == totals[1]


def main():
    tests = [
        test_family_size_and_splits,
        test_condition_ids_do_not_wrap,
        test_first_256_structural_specs_are_unique,
        test_pair_becomes_sensor_identical_after_cue_hides,
        test_teacher_solves_both_counterfactual_members,
    ]
    for fn in tests:
        fn()
        print("PASS:", fn.__name__)
    print("ALL_VARIED_CUED_DETOUR_TESTS_PASSED")


if __name__ == "__main__":
    main()
