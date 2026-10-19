#!/usr/bin/env python3
"""
Structural audit for LateInstructionDetourWorld v1.

Audits all 1600 TRAIN structural conditions without neural inference.

Guarantees:
- instruction arrives strictly after cue erasure and before the first lateral
  commitment;
- NORMAL/INVERT is absent from ordinary sensory observation/render channels;
- cue counterfactuals become sensory-identical after cue erasure under matched
  pre-decision actions;
- response mapping is exactly cue XOR instruction;
- old DEV/TEST IDs are rejected by the qualification environment.
"""
from __future__ import annotations

import numpy as np

from continuous_brain.late_instruction_detour_world_v1 import (
    INSTRUCTION_INVERT,
    INSTRUCTION_NONE,
    INSTRUCTION_NORMAL,
    LateInstructionDetourWorld,
)
from continuous_brain.varied_cued_detour_world import (
    LEFT,
    RIGHT,
    TRAIN_CONDITION_RANGE,
    UP,
)


def same(a, b):
    return bool(np.array_equal(a, b))


def audit_condition(cid):
    # Instruction must be absent from sensory channels.
    for cue in ("left", "right"):
        n = LateInstructionDetourWorld(
            cid,
            cue=cue,
            instruction="normal",
            instruction_delay=1,
        )
        i = LateInstructionDetourWorld(
            cid,
            cue=cue,
            instruction="invert",
            instruction_delay=1,
        )
        on = n.reset()
        oi = i.reset()

        if not same(on, oi):
            raise AssertionError("instruction leaked into reset observation")

        first_lateral = None
        while not n.done and not i.done:
            if n.steps != i.steps:
                raise AssertionError("instruction pair step mismatch")
            if n.instruction_id() != (
                INSTRUCTION_NONE
                if n.steps < n.instruction_start_step
                else INSTRUCTION_NORMAL
            ):
                raise AssertionError("NORMAL instruction schedule mismatch")
            if i.instruction_id() != (
                INSTRUCTION_NONE
                if i.steps < i.instruction_start_step
                else INSTRUCTION_INVERT
            ):
                raise AssertionError("INVERT instruction schedule mismatch")

            if not same(n._obs(), i._obs()):
                raise AssertionError(
                    "instruction leaked into ordinary observation"
                )

            # Render check on a bounded subset is performed by caller.
            tn = int(n.teacher_action())
            ti = int(i.teacher_action())
            if tn in (LEFT, RIGHT) or ti in (LEFT, RIGHT):
                first_lateral = int(n.steps)
                if not n.instruction_visible() or not i.instruction_visible():
                    raise AssertionError(
                        "decision occurred before instruction arrival"
                    )
                if tn == ti:
                    raise AssertionError(
                        "NORMAL/INVERT failed to reverse first commitment"
                    )
                break

            if tn != UP or ti != UP:
                raise AssertionError(
                    "pre-decision instruction changed primitive policy"
                )

            on, _, _, _ = n.step(UP)
            oi, _, _, _ = i.step(UP)
            if not same(on, oi):
                raise AssertionError(
                    "instruction leaked through matched physical transition"
                )

        if first_lateral is None:
            raise AssertionError("no lateral commitment found")
        if not (n.cue_steps < n.instruction_start_step < first_lateral):
            raise AssertionError(
                "instruction timing is not strictly post-cue/pre-decision"
            )

    # Cue pair must be identical after cue erasure under identical actions.
    l = LateInstructionDetourWorld(
        cid,
        cue="left",
        instruction="normal",
        instruction_delay=1,
    )
    r = LateInstructionDetourWorld(
        cid,
        cue="right",
        instruction="normal",
        instruction_delay=1,
    )
    l.reset()
    r.reset()

    while l.steps < l.cue_steps:
        l.step(UP)
        r.step(UP)

    if not same(l._obs(), r._obs()):
        raise AssertionError("cue remained in observation after erasure")

    while True:
        tl = int(l.teacher_action())
        tr = int(r.teacher_action())
        if tl in (LEFT, RIGHT) or tr in (LEFT, RIGHT):
            if tl == tr:
                raise AssertionError("cue failed to determine NORMAL response")
            break
        if not same(l._obs(), r._obs()):
            raise AssertionError(
                "post-cue pair diverged before cue-dependent decision"
            )
        l.step(UP)
        r.step(UP)

    # Exact response mapping.
    mapping = {}
    for cue in ("left", "right"):
        for instruction in ("normal", "invert"):
            e = LateInstructionDetourWorld(
                cid,
                cue=cue,
                instruction=instruction,
                instruction_delay=1,
            )
            e.reset()
            mapping[(cue, instruction)] = e.response_side

    expected = {
        ("left", "normal"): "left",
        ("right", "normal"): "right",
        ("left", "invert"): "right",
        ("right", "invert"): "left",
    }
    if mapping != expected:
        raise AssertionError(
            f"cue×instruction response mapping mismatch: {mapping}"
        )


def main():
    for idx, cid in enumerate(TRAIN_CONDITION_RANGE, 1):
        audit_condition(int(cid))

        # Visual channel audit on a representative deterministic subset.
        if int(cid) < 64:
            for cue in ("left", "right"):
                n = LateInstructionDetourWorld(
                    cid,
                    cue=cue,
                    instruction="normal",
                    instruction_delay=1,
                )
                i = LateInstructionDetourWorld(
                    cid,
                    cue=cue,
                    instruction="invert",
                    instruction_delay=1,
                )
                n.reset()
                i.reset()
                while True:
                    if not np.array_equal(
                        n.render_rgb(96),
                        i.render_rgb(96),
                    ):
                        raise AssertionError(
                            "instruction leaked into image channel"
                        )
                    t = int(n.teacher_action())
                    if t in (LEFT, RIGHT):
                        break
                    n.step(UP)
                    i.step(UP)

        if idx == 1 or idx % 200 == 0 or idx == len(TRAIN_CONDITION_RANGE):
            print(f"audited={idx}/{len(TRAIN_CONDITION_RANGE)}")

    # Qualification environment must reject already-opened old held-out IDs.
    for cid in (1600, 1979, 1980, 2359):
        rejected = False
        try:
            LateInstructionDetourWorld(
                cid,
                cue="left",
                instruction="normal",
            )
        except ValueError:
            rejected = True
        if not rejected:
            raise AssertionError(
                f"qualification environment accepted held-out ID {cid}"
            )

    print("instruction_absent_from_observation: true")
    print("instruction_absent_from_render: true")
    print("instruction_strictly_post_cue_pre_decision: true")
    print("cue_pair_post_erasure_identical: true")
    print("four_way_response_mapping_exact: true")
    print("OLD_DEV_IDS_REJECTED: true")
    print("OLD_TEST_IDS_REJECTED: true")
    print("LATE_INSTRUCTION_TASK_AUDIT_V1: PASS")


if __name__ == "__main__":
    main()
