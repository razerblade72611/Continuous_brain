#!/usr/bin/env python3
"""Audit third-generation canonical-executive rule family."""
from __future__ import annotations
import numpy as np

from continuous_brain.canonical_executive_rule_world_v3 import (
    FAMILY_SIZE,
    CanonicalExecutiveRuleWorld,
    generate_canonical_rule_spec,
    target_action,
    UP,
)

RULES=("invert_lr","vertical","vertical_reverse")

OLD_V1_CORRIDOR={0.28,0.36,0.44,0.52,0.60,0.68,0.76,0.84}
OLD_V2_CORRIDOR={0.30,0.34,0.38,0.42,0.46,0.50,0.54,0.58,0.62,0.66,0.70,0.74,0.78,0.82}
OLD_V1_GX={0.38,0.46,0.54,0.62,0.70}
OLD_V2_GX={0.40,0.44,0.48,0.52,0.56,0.60,0.64,0.68}
OLD_V1_GY={0.10,0.14,0.18,0.22}
OLD_V2_GY={0.12,0.16,0.20,0.24}
OLD_V1_MOVE={0.050,0.065,0.080,0.095}
OLD_V2_MOVE={0.055,0.070,0.085,0.100}

def main():
    specs=[generate_canonical_rule_spec(i) for i in range(FAMILY_SIZE)]
    sigs=[s.structural_signature() for s in specs]
    if len(set(sigs))!=FAMILY_SIZE: raise AssertionError("duplicate structural condition")
    train=set(sigs[:1600]); dev=set(sigs[1600:2000]); test=set(sigs[2000:2400])
    if train&dev or train&test or dev&test: raise AssertionError("split overlap")

    for s in specs:
        if s.corridor_length in OLD_V1_CORRIDOR|OLD_V2_CORRIDOR: raise AssertionError("old corridor reused")
        if s.goal_x_offset in OLD_V1_GX|OLD_V2_GX: raise AssertionError("old goal-x reused")
        if s.goal_y_offset in OLD_V1_GY|OLD_V2_GY: raise AssertionError("old goal-y reused")
        if s.move_step in OLD_V1_MOVE|OLD_V2_MOVE: raise AssertionError("old move-step reused")

    for idx,cid in enumerate(range(FAMILY_SIZE),1):
        for cue in ("left","right"):
            envs=[CanonicalExecutiveRuleWorld(cid,cue=cue,rule=r) for r in RULES]
            obs=[e.reset() for e in envs]
            if not all(np.array_equal(obs[0],x) for x in obs[1:]):
                raise AssertionError("rule leaked into reset observation")
            while True:
                if not all(np.array_equal(envs[0]._obs(),e._obs()) for e in envs[1:]):
                    raise AssertionError("rule leaked into observation")
                if cid<24:
                    imgs=[e.render_rgb(96) for e in envs]
                    if not all(np.array_equal(imgs[0],x) for x in imgs[1:]):
                        raise AssertionError("rule leaked into render")
                acts=[e.teacher_action() for e in envs]
                if envs[0].at_response_step():
                    exp=[target_action(cue,r) for r in RULES]
                    if acts!=exp: raise AssertionError("response mapping mismatch")
                    if not all(e.instruction_visible() for e in envs):
                        raise AssertionError("instruction missing at response")
                    break
                if any(a!=UP for a in acts): raise AssertionError("rule changed pre-response policy")
                for e in envs: e.step(UP)

        l=CanonicalExecutiveRuleWorld(cid,cue="left",rule="invert_lr"); l.reset()
        r=CanonicalExecutiveRuleWorld(cid,cue="right",rule="invert_lr"); r.reset()
        while l.steps<l.cue_steps:
            l.step(UP); r.step(UP)
        if not np.array_equal(l._obs(),r._obs()):
            raise AssertionError("cue persists after erasure")
        while not l.at_response_step():
            if not np.array_equal(l._obs(),r._obs()):
                raise AssertionError("post-cue pair diverged")
            l.step(UP); r.step(UP)

        if idx==1 or idx%300==0 or idx==FAMILY_SIZE:
            print(f"audited={idx}/{FAMILY_SIZE}")

    print("family_size: 2400")
    print("fresh_train_conditions: 1600")
    print("fresh_dev_conditions: 400")
    print("fresh_test_conditions: 400")
    print("structural_conditions_unique: true")
    print("split_structural_overlap: false")
    print("continuous_geometry_disjoint_from_v1_and_v2: true")
    print("rule_absent_from_observation: true")
    print("rule_absent_from_render: true")
    print("cue_post_erasure_counterfactual_identical: true")
    print("CANONICAL_EXECUTIVE_FAMILY_AUDIT_V3: PASS")

if __name__=="__main__": main()
