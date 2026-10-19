#!/usr/bin/env python3
from __future__ import annotations

import argparse, json
from pathlib import Path
import torch

from continuous_brain.memory_guided_goal_option_v1 import (
    BASE_CHECKPOINT, MEMORY_CHECKPOINT, assert_pair_identical,
    build_substrate, collect_decision_snapshot, option_table, sha256,
)
from continuous_brain.varied_cued_detour_world import VariedCuedDetourWorld, LEFT, RIGHT
from continuous_brain.evaluate_competitive_infant import _context_tensor, _image_tensor
from continuous_brain.planning_v103 import _active_option_should_terminate, _option_policy_action

DEV_IDS = range(1600, 1728)
MAX_STEPS = 160
MIN_COMMIT = 3
TERMINATE_THRESHOLD = .50
SUPPORT_THRESHOLD = .90

def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0

@torch.no_grad()
def execute_option(base, cfg, snap, option):
    env = VariedCuedDetourWorld(
        int(snap["condition_id"]), cue=str(snap["cue"]),
        max_episode_steps=MAX_STEPS, image_size_hint=int(cfg.image_size),
    )
    obs = env.reset()
    for action in snap["prefix_actions"]:
        obs, _, done, _ = env.step(int(action))
        if done:
            raise RuntimeError("prefix replay ended episode")

    expected = LEFT if snap["cue"] == "left" else RIGHT
    if int(env.teacher_action()) != int(expected):
        raise AssertionError("prefix did not reach delayed decision")
    if env.steps < env.cue_steps:
        raise AssertionError("cue still visible at delayed decision")

    current = int(option)
    age = 0
    initial_active = True
    initial_steps = 0
    first_action = None
    first_lateral = None
    first_lateral_step = None
    terminated_initial = False
    total_reward = 0.0
    info = {"reached_correct": False}

    while not env.done:
        image = _image_tensor(env, cfg.image_size, "cpu")
        _, context_dyn = _context_tensor(obs, "cpu", cfg)
        spatial = base.target_spatial_bus(image)
        root = base.planning.encode_state(spatial, context_dyn)

        if _active_option_should_terminate(
            base, root, current, age, MIN_COMMIT, TERMINATE_THRESHOLD
        ):
            if initial_active:
                initial_active = False
                terminated_initial = True
            selector = base.options.selector_logits(root)[0]
            current = int(torch.argmax(selector).item())
            age = 0

        action = int(_option_policy_action(base, root, current))
        if first_action is None:
            first_action = action
        if initial_active:
            initial_steps += 1
        if first_lateral is None and action in (LEFT, RIGHT):
            first_lateral = action
            first_lateral_step = int(env.steps - len(snap["prefix_actions"]) + 1)

        obs, reward, _, info = env.step(action)
        total_reward += float(reward)
        age += 1

    return {
        "option": int(option),
        "first_action": first_action,
        "immediate_match": bool(first_action == expected),
        "first_lateral": first_lateral,
        "first_lateral_step": first_lateral_step,
        "temporal_commitment_match": bool(first_lateral == expected),
        "completed_correct": bool(info.get("reached_correct", False)),
        "initial_option_steps": int(initial_steps),
        "initial_option_terminated": bool(terminated_initial),
        "collisions": int(getattr(env, "collisions", 0)),
        "reward": float(total_reward),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="state/temporal_option_affordance_v1/result.json")
    args = ap.parse_args()

    before_base = sha256(BASE_CHECKPOINT)
    before_mem = sha256(MEMORY_CHECKPOINT)

    print("TEMPORAL_OPTION_AFFORDANCE_V1")
    print("device: cpu")
    print("training_performed: false")
    print("development_window_reused: 1600-1727")
    print("new_development_window_consumed: false")
    print("test_touched: false")
    print("detour_planner_repair_closed: true")
    print("all_options_enumerated: true")

    base, cfg, admitted = build_substrate("cpu")

    members = []
    any_immediate, any_temporal, any_completion = [], [], []
    top1_immediate, top1_temporal = [], []
    paired_temporal, paired_completion = [], []

    for n, cid in enumerate(DEV_IDS, start=1):
        left = collect_decision_snapshot(base, cfg, admitted, cid, "left", device="cpu")
        right = collect_decision_snapshot(base, cfg, admitted, cid, "right", device="cpu")
        assert_pair_identical(left, right)

        ct, cc = [], []
        for snap in (left, right):
            table = option_table(base, snap["root"])
            top1 = max(table, key=lambda r: r["selector_logit"])["option"]
            outcomes = [execute_option(base, cfg, snap, o) for o in range(int(base.cfg.option_count))]
            immediate = [o["option"] for o in outcomes if o["immediate_match"]]
            temporal = [o["option"] for o in outcomes if o["temporal_commitment_match"]]
            completion = [o["option"] for o in outcomes if o["completed_correct"]]
            top = next(o for o in outcomes if o["option"] == top1)

            any_immediate.append(float(bool(immediate)))
            any_temporal.append(float(bool(temporal)))
            any_completion.append(float(bool(completion)))
            top1_immediate.append(float(top["immediate_match"]))
            top1_temporal.append(float(top["temporal_commitment_match"]))
            ct.append(bool(temporal))
            cc.append(bool(completion))

            members.append({
                "condition_id": int(cid), "cue": snap["cue"], "selector_top1": int(top1),
                "immediate_options": immediate, "temporal_options": temporal,
                "completion_options": completion, "outcomes": outcomes,
            })

        paired_temporal.append(float(all(ct)))
        paired_completion.append(float(all(cc)))
        if n == 1 or n % 8 == 0:
            print(f"diagnostic_conditions={n}/{len(DEV_IDS)}")

    summary = {
        "members": len(members),
        "conditions": len(DEV_IDS),
        "any_immediate_goal_option_rate": mean(any_immediate),
        "any_temporal_goal_option_rate": mean(any_temporal),
        "any_correct_completion_option_rate": mean(any_completion),
        "both_sides_temporal_support_condition_rate": mean(paired_temporal),
        "both_sides_completion_support_condition_rate": mean(paired_completion),
        "selector_top1_immediate_commitment_rate": mean(top1_immediate),
        "selector_top1_temporal_commitment_rate": mean(top1_temporal),
    }

    if (summary["any_temporal_goal_option_rate"] >= SUPPORT_THRESHOLD and
        summary["both_sides_temporal_support_condition_rate"] >= SUPPORT_THRESHOLD):
        finding = "TEMPORAL_OPTION_REPERTOIRE_SUFFICIENT_FOR_GOAL_SELECTOR"
        next_step = "Train a memory/goal-conditioned option selector; keep memory and primitive option policies frozen."
    else:
        finding = "TEMPORAL_OPTION_REPERTOIRE_INSUFFICIENT"
        next_step = "Do not train another selector; expand/relearn goal-diverse temporal options while preserving admitted memory."

    if sha256(BASE_CHECKPOINT) != before_base or sha256(MEMORY_CHECKPOINT) != before_mem:
        raise RuntimeError("protected checkpoint changed")

    result = {
        "format": "continuous_brain.temporal_option_affordance_v1",
        "training_performed": False, "development_window_reused": [1600, 1727],
        "new_development_window_consumed": False, "test_touched": False,
        "summary": summary, "finding": finding, "next_step": next_step,
        "members": members, "protected_unchanged": True,
        "v019_remains_closed": True, "detour_planner_repair_closed": True,
    }
    p = Path(args.output)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2, sort_keys=True))

    print("\nSUMMARY")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print("finding:", finding)
    print("next_step:", next_step)
    print("PROTECTED_UNCHANGED: true")
    print("TRAINING_PERFORMED: false")
    print("TEST_TOUCHED: false")
    print("V019_REMAINS_CLOSED: true")
    print("DETOUR_PLANNER_REPAIR_CLOSED: true")
    print("TEMPORAL_OPTION_AFFORDANCE_DIAGNOSTIC: PASS")
    print("TEMPORAL_OPTION_AFFORDANCE_COMPLETE")
    print("result:", p)

if __name__ == "__main__":
    main()
