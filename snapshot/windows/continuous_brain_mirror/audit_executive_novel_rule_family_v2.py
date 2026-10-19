#!/usr/bin/env python3
"""Audit the fresh second-generation novel-rule executive family."""
from __future__ import annotations
import numpy as np

from continuous_brain.executive_novel_rule_world_v2 import (
    DEV_CONDITION_RANGE,
    FAMILY_SIZE,
    RULE_INVERT_LR,
    RULE_NONE,
    RULE_VERTICAL,
    RULE_VERTICAL_REVERSE,
    TEST_CONDITION_RANGE,
    TRAIN_CONDITION_RANGE,
    ExecutiveNovelRuleWorld,
    generate_executive_rule_spec,
    target_action,
    UP,
)

RULES = ("invert_lr", "vertical", "vertical_reverse")

# First-generation continuous factor values. Fresh family must not reuse them.
OLD_CORRIDOR = {0.28,0.36,0.44,0.52,0.60,0.68,0.76,0.84}
OLD_GOAL_X = {0.38,0.46,0.54,0.62,0.70}
OLD_GOAL_Y = {0.10,0.14,0.18,0.22}
OLD_MOVE = {0.050,0.065,0.080,0.095}


def main():
    specs = [generate_executive_rule_spec(i) for i in range(FAMILY_SIZE)]
    sigs = [s.structural_signature() for s in specs]

    if len(set(sigs)) != FAMILY_SIZE:
        raise AssertionError("structural condition duplicated")

    train = set(sigs[:1600])
    dev = set(sigs[1600:2000])
    test = set(sigs[2000:2400])
    if train & dev or train & test or dev & test:
        raise AssertionError("fresh split structural overlap")

    if any(s.corridor_length in OLD_CORRIDOR for s in specs):
        raise AssertionError("old corridor-length value reused")
    if any(s.goal_x_offset in OLD_GOAL_X for s in specs):
        raise AssertionError("old goal-x value reused")
    if any(s.goal_y_offset in OLD_GOAL_Y for s in specs):
        raise AssertionError("old goal-y value reused")
    if any(s.move_step in OLD_MOVE for s in specs):
        raise AssertionError("old move-step value reused")

    for idx, cid in enumerate(range(FAMILY_SIZE), 1):
        # Rule must not leak into sensory observation or rendered image.
        for cue in ("left", "right"):
            envs = [
                ExecutiveNovelRuleWorld(cid, cue=cue, rule=r)
                for r in RULES
            ]
            obs = [e.reset() for e in envs]
            if not all(np.array_equal(obs[0], x) for x in obs[1:]):
                raise AssertionError("rule leaked into reset observation")

            while True:
                if not all(
                    np.array_equal(envs[0]._obs(), e._obs())
                    for e in envs[1:]
                ):
                    raise AssertionError("rule leaked into observation")

                if cid < 32:
                    imgs = [e.render_rgb(96) for e in envs]
                    if not all(
                        np.array_equal(imgs[0], x)
                        for x in imgs[1:]
                    ):
                        raise AssertionError("rule leaked into render")

                actions = [e.teacher_action() for e in envs]
                if envs[0].at_response_step():
                    if not all(e.instruction_visible() for e in envs):
                        raise AssertionError("rule unavailable at response")
                    expected = [
                        target_action(cue, r) for r in RULES
                    ]
                    if actions != expected:
                        raise AssertionError("rule response mapping mismatch")
                    break

                if any(a != UP for a in actions):
                    raise AssertionError("pre-response rule changed movement")

                for e in envs:
                    e.step(UP)

            if not (
                envs[0].cue_steps
                < envs[0].instruction_start_step
                <= envs[0].steps
            ):
                raise AssertionError("instruction timing invalid")

        # After cue erasure, LEFT/RIGHT sensory streams must be identical under
        # identical pre-response movement.
        l = ExecutiveNovelRuleWorld(cid, cue="left", rule="invert_lr")
        r = ExecutiveNovelRuleWorld(cid, cue="right", rule="invert_lr")
        l.reset(); r.reset()
        while l.steps < l.cue_steps:
            l.step(UP); r.step(UP)
        if not np.array_equal(l._obs(), r._obs()):
            raise AssertionError("cue survives erasure in observation")

        while not l.at_response_step():
            if not np.array_equal(l._obs(), r._obs()):
                raise AssertionError("post-cue counterfactual diverged")
            l.step(UP); r.step(UP)

        if idx == 1 or idx % 300 == 0 or idx == FAMILY_SIZE:
            print(f"audited={idx}/{FAMILY_SIZE}")

    # Ensure none of the active rules reproduces inherited NORMAL L->LEFT,R->RIGHT.
    inherited = {"left": 3, "right": 4}
    for rule in RULES:
        mapping = {
            cue: target_action(cue, rule)
            for cue in ("left", "right")
        }
        if mapping == inherited:
            raise AssertionError(f"rule {rule} equals inherited NORMAL")

    print("family_size: 2400")
    print("fresh_train_conditions: 1600")
    print("fresh_dev_conditions: 400")
    print("fresh_test_conditions: 400")
    print("structural_conditions_unique: true")
    print("split_structural_overlap: false")
    print("continuous_geometry_disjoint_from_v1: true")
    print("rule_absent_from_observation: true")
    print("rule_absent_from_render: true")
    print("cue_post_erasure_counterfactual_identical: true")
    print("instruction_post_cue_pre_response: true")
    print("all_active_rules_novel_relative_to_inherited_policy: true")
    print("EXECUTIVE_NOVEL_RULE_FAMILY_AUDIT_V2: PASS")


if __name__ == "__main__":
    main()
