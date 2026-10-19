#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from continuous_brain.episodic_goal_skill_integration_v1 import (
    EPISODIC_READER_PATH,
    EPISODIC_TEST_RESULT,
    EXPECTED_EPISODIC_READER_SHA,
    EXPECTED_GOAL_DECODER_SHA,
    EXPECTED_GOAL_OPTIONS_SHA,
    EXPECTED_TEMPORAL_CONTROLLER_SHA,
    GOAL_DECODER_PATH,
    GOAL_OPTIONS_PATH,
    TEMPORAL_CONTROLLER_PATH,
    decode_goal,
    select_canonical_recall,
    verify_integration_provenance,
)
from continuous_brain.additive_goal_options_v1 import (
    load_goal_options,
    sha256,
)
from continuous_brain.recurrent_goal_option_state_v1 import (
    load_controller,
)
from continuous_brain.multi_item_episodic_memory_v1 import (
    QUERY_FIRST,
    QUERY_SECOND,
    load_reader,
)
from continuous_brain.memory_guided_goal_option_v1 import (
    BASE_CHECKPOINT,
    MEMORY_CHECKPOINT,
    assert_pair_identical,
    build_substrate,
    collect_decision_snapshot,
    load_goal_decoder,
)
from train_test_recurrent_goal_option_state_v1 import (
    run_baseline,
    run_goal_skill,
)

CASE_IDS = list(range(64))
SOURCE_FAMILY_SIZE = 1600

MIN_ATTENTION_ROUTE = .99
MIN_SELECTED_GOAL = .99
MIN_COMMITMENT = .95
MIN_PAIR_BOTH = .95
MIN_QUERY_FLIP = .95
MIN_SWAP = .95
MIN_DONOR = .95
MIN_NONTARGET_PRESERVE = .95
MIN_SAME_PAIR_STABILITY = .99
MIN_COMMITMENT_ADVANTAGE = .25
MAX_BASELINE_COMMITMENT = .70


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def cue_value(cue):
    return 0 if cue == "left" else 1


def cue_name(value):
    return "left" if int(value) == 0 else "right"


def distinct_mapped_id(seed, mul, add, avoid):
    out = (int(seed) * int(mul) + int(add)) % SOURCE_FAMILY_SIZE
    while out in avoid:
        out = (out + 1) % SOURCE_FAMILY_SIZE
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--state-dir",
        default="state/episodic_goal_skill_integration_v1",
    )
    args = ap.parse_args()

    state = Path(args.state_dir)
    state.mkdir(parents=True, exist_ok=True)
    result_path = state / "result.json"

    if not EPISODIC_TEST_RESULT.is_file():
        raise FileNotFoundError(EPISODIC_TEST_RESULT)

    episodic_test = json.loads(
        EPISODIC_TEST_RESULT.read_text()
    )
    if episodic_test.get("test_admission_passed") is not True:
        raise RuntimeError(
            "v0.23.1 frozen composition TEST did not pass"
        )
    if episodic_test.get("composition_test_spent") is not True:
        raise RuntimeError(
            "v0.23.1 TEST result missing spent-test seal"
        )

    provenance = verify_integration_provenance()

    protected_before = {
        "base": sha256(BASE_CHECKPOINT),
        "memory": sha256(MEMORY_CHECKPOINT),
        "goal_decoder": sha256(GOAL_DECODER_PATH),
        "goal_options": sha256(GOAL_OPTIONS_PATH),
        "temporal_controller":
            sha256(TEMPORAL_CONTROLLER_PATH),
        "episodic_reader": sha256(EPISODIC_READER_PATH),
        "episodic_test_result":
            sha256(EPISODIC_TEST_RESULT),
    }

    print("EPISODIC_GOAL_SKILL_INTEGRATION_V1")
    print("device: cpu")
    print("training_performed: false")
    print("new_holdout_consumed: false")
    print("source_conditions: historical TRAIN family only")
    print("integration_cases: 64")
    print("event_pair_combinations_per_case: 4")
    print("late_queries: FIRST, SECOND")
    print("relation_queries_preserved_but_not_motor_mapped: SAME, DIFFERENT")
    print("v023_test_admission_required: true")
    print("v023_composition_test_spent: true")
    print("v0222_formal_status_preserved: FAIL")
    print("v0222_temporal_controller_reused_as_best_preserved_mechanism: true")

    base, cfg, admitted = build_substrate("cpu")
    decoder, _ = load_goal_decoder(
        GOAL_DECODER_PATH, "cpu"
    )
    bank, _ = load_goal_options(
        GOAL_OPTIONS_PATH,
        EXPECTED_GOAL_DECODER_SHA,
        "cpu",
    )
    controller, _ = load_controller(
        TEMPORAL_CONTROLLER_PATH,
        expected_goal_options_sha256=
            EXPECTED_GOAL_OPTIONS_SHA,
        expected_goal_decoder_sha256=
            EXPECTED_GOAL_DECODER_SHA,
        device="cpu",
    )
    reader, _ = load_reader(
        EPISODIC_READER_PATH,
        "cpu",
    )

    recall_cache = {}
    execution_cache = {}
    baseline_cache = {}
    exec_prefix_cache = {}

    def get_recall(cid, cue):
        key = (int(cid), str(cue))
        if key not in recall_cache:
            snap = collect_decision_snapshot(
                base,
                cfg,
                admitted,
                int(cid),
                str(cue),
                device="cpu",
            )
            recall_cache[key] = snap["recall"].clone()
        return recall_cache[key]

    def get_exec_prefix(exec_id):
        exec_id = int(exec_id)
        if exec_id not in exec_prefix_cache:
            left = collect_decision_snapshot(
                base, cfg, admitted, exec_id, "left"
            )
            right = collect_decision_snapshot(
                base, cfg, admitted, exec_id, "right"
            )
            assert_pair_identical(left, right)
            if left["prefix_actions"] != right["prefix_actions"]:
                raise AssertionError(
                    "execution prefixes differ by hidden target"
                )
            exec_prefix_cache[exec_id] = list(
                left["prefix_actions"]
            )
        return exec_prefix_cache[exec_id]

    def execute_goal(exec_id, goal_side):
        key = (int(exec_id), int(goal_side))
        if key not in execution_cache:
            prefix = get_exec_prefix(exec_id)
            execution_cache[key] = run_goal_skill(
                base,
                cfg,
                bank,
                controller,
                cid=int(exec_id),
                actual_cue=cue_name(goal_side),
                prefix=prefix,
                goal_side=int(goal_side),
            )
        return execution_cache[key]

    def execute_baseline(exec_id, target_side):
        key = (int(exec_id), int(target_side))
        if key not in baseline_cache:
            prefix = get_exec_prefix(exec_id)
            baseline_cache[key] = run_baseline(
                base,
                cfg,
                int(exec_id),
                cue_name(target_side),
                prefix,
            )
        return baseline_cache[key]

    attention_correct = []
    goal_correct = []
    commitment_correct = []
    completion_correct = []
    baseline_commitment = []
    baseline_completion = []
    pair_both_queries = []

    unequal_query_goal_flip = []
    unequal_query_action_flip = []
    unequal_swap_goal_flip = []
    unequal_swap_action_flip = []

    donor0_target_goal = []
    donor0_target_action = []
    donor0_nontarget_goal = []
    donor0_nontarget_action = []

    donor1_target_goal = []
    donor1_target_action = []
    donor1_nontarget_goal = []
    donor1_nontarget_action = []

    same_pair_goal_stability = []
    same_pair_action_stability = []

    details = []

    for n, case_id in enumerate(CASE_IDS, start=1):
        e1 = int(case_id)
        e2 = distinct_mapped_id(
            case_id, 37, 401, {e1}
        )
        exec_id = distinct_mapped_id(
            case_id, 73, 907, {e1, e2}
        )
        donor_id = distinct_mapped_id(
            case_id, 53, 211, {e1, e2, exec_id}
        )

        get_exec_prefix(exec_id)

        for cue1 in ("left", "right"):
            for cue2 in ("left", "right"):
                s0 = get_recall(e1, cue1)
                s1 = get_recall(e2, cue2)

                normal = {}
                member_ok = []

                for query, expected_slot, expected_cue in (
                    (QUERY_FIRST, 0, cue1),
                    (QUERY_SECOND, 1, cue2),
                ):
                    sel = select_canonical_recall(
                        reader, s0, s1, query
                    )
                    attention_slot = int(
                        torch.argmax(
                            sel["attention"],
                            dim=-1,
                        )[0].item()
                    )
                    goal, goal_logits = decode_goal(
                        decoder,
                        sel["selected_recall"],
                    )
                    expected_goal = cue_value(expected_cue)

                    run = (
                        execute_goal(exec_id, goal)
                        if goal is not None
                        else None
                    )
                    baseline = execute_baseline(
                        exec_id, expected_goal
                    )

                    att_ok = attention_slot == expected_slot
                    goal_ok = goal == expected_goal
                    action_ok = (
                        run is not None
                        and bool(run["commitment_correct"])
                    )

                    attention_correct.append(float(att_ok))
                    goal_correct.append(float(goal_ok))
                    commitment_correct.append(float(action_ok))
                    completion_correct.append(
                        float(
                            run is not None
                            and run["completed_correct"]
                        )
                    )
                    baseline_commitment.append(
                        float(baseline["commitment_correct"])
                    )
                    baseline_completion.append(
                        float(baseline["completed_correct"])
                    )

                    member_ok.append(
                        bool(att_ok and goal_ok and action_ok)
                    )
                    normal[query] = {
                        "expected_slot": expected_slot,
                        "expected_goal": expected_goal,
                        "attention_slot": attention_slot,
                        "attention": [
                            float(x)
                            for x in sel["attention"][0]
                        ],
                        "goal": goal,
                        "goal_logits": [
                            float(x)
                            for x in goal_logits[0]
                        ],
                        "first_lateral":
                            None if run is None
                            else run["first_lateral"],
                        "completed":
                            False if run is None
                            else bool(run["completed_correct"]),
                    }

                pair_both_queries.append(
                    float(all(member_ok))
                )

                v1 = cue_value(cue1)
                v2 = cue_value(cue2)

                if v1 != v2:
                    first = normal[QUERY_FIRST]
                    second = normal[QUERY_SECOND]

                    unequal_query_goal_flip.append(
                        float(first["goal"] != second["goal"])
                    )
                    unequal_query_action_flip.append(
                        float(
                            first["first_lateral"]
                            != second["first_lateral"]
                        )
                    )

                    # Slot swap: FIRST should now retrieve former SECOND.
                    swap_first = select_canonical_recall(
                        reader, s1, s0, QUERY_FIRST
                    )
                    swap_goal, _ = decode_goal(
                        decoder,
                        swap_first["selected_recall"],
                    )
                    swap_run = execute_goal(
                        exec_id, swap_goal
                    )
                    unequal_swap_goal_flip.append(
                        float(swap_goal == v2)
                    )
                    unequal_swap_action_flip.append(
                        float(
                            swap_run["first_lateral"]
                            != first["first_lateral"]
                        )
                    )

                    # Donor-replace slot0 with opposite content.
                    donor0_cue = (
                        "right" if cue1 == "left" else "left"
                    )
                    d0 = get_recall(donor_id, donor0_cue)

                    d0_first = select_canonical_recall(
                        reader, d0, s1, QUERY_FIRST
                    )
                    d0g_first, _ = decode_goal(
                        decoder,
                        d0_first["selected_recall"],
                    )
                    d0r_first = execute_goal(
                        exec_id, d0g_first
                    )
                    donor0_target_goal.append(
                        float(
                            d0g_first
                            == cue_value(donor0_cue)
                        )
                    )
                    donor0_target_action.append(
                        float(
                            d0r_first[
                                "commitment_follows_goal"
                            ]
                        )
                    )

                    d0_second = select_canonical_recall(
                        reader, d0, s1, QUERY_SECOND
                    )
                    d0g_second, _ = decode_goal(
                        decoder,
                        d0_second["selected_recall"],
                    )
                    d0r_second = execute_goal(
                        exec_id, d0g_second
                    )
                    donor0_nontarget_goal.append(
                        float(d0g_second == v2)
                    )
                    donor0_nontarget_action.append(
                        float(
                            d0r_second[
                                "commitment_follows_goal"
                            ]
                        )
                    )

                    # Donor-replace slot1 with opposite content.
                    donor1_cue = (
                        "right" if cue2 == "left" else "left"
                    )
                    d1 = get_recall(donor_id, donor1_cue)

                    d1_second = select_canonical_recall(
                        reader, s0, d1, QUERY_SECOND
                    )
                    d1g_second, _ = decode_goal(
                        decoder,
                        d1_second["selected_recall"],
                    )
                    d1r_second = execute_goal(
                        exec_id, d1g_second
                    )
                    donor1_target_goal.append(
                        float(
                            d1g_second
                            == cue_value(donor1_cue)
                        )
                    )
                    donor1_target_action.append(
                        float(
                            d1r_second[
                                "commitment_follows_goal"
                            ]
                        )
                    )

                    d1_first = select_canonical_recall(
                        reader, s0, d1, QUERY_FIRST
                    )
                    d1g_first, _ = decode_goal(
                        decoder,
                        d1_first["selected_recall"],
                    )
                    d1r_first = execute_goal(
                        exec_id, d1g_first
                    )
                    donor1_nontarget_goal.append(
                        float(d1g_first == v1)
                    )
                    donor1_nontarget_action.append(
                        float(
                            d1r_first[
                                "commitment_follows_goal"
                            ]
                        )
                    )
                else:
                    first = normal[QUERY_FIRST]
                    second = normal[QUERY_SECOND]
                    same_pair_goal_stability.append(
                        float(first["goal"] == second["goal"])
                    )
                    same_pair_action_stability.append(
                        float(
                            first["first_lateral"]
                            == second["first_lateral"]
                        )
                    )

                details.append(
                    {
                        "case_id": case_id,
                        "event1_id": e1,
                        "event2_id": e2,
                        "execution_id": exec_id,
                        "donor_id": donor_id,
                        "cue1": cue1,
                        "cue2": cue2,
                        "normal": {
                            "first": normal[QUERY_FIRST],
                            "second": normal[QUERY_SECOND],
                        },
                    }
                )

        if n == 1 or n % 8 == 0:
            print(
                f"integration_cases={n}/{len(CASE_IDS)}"
            )

    own_commit = mean(commitment_correct)
    base_commit = mean(baseline_commitment)

    metrics = {
        "members": len(commitment_correct),
        "pair_members": len(pair_both_queries),
        "attention_route_accuracy":
            mean(attention_correct),
        "selected_goal_accuracy":
            mean(goal_correct),
        "autonomous_commitment_correct":
            own_commit,
        "pair_both_queries_causal_chain":
            mean(pair_both_queries),
        "autonomous_completion_correct":
            mean(completion_correct),
        "baseline_commitment_correct":
            base_commit,
        "baseline_completion_correct":
            mean(baseline_completion),
        "commitment_advantage":
            own_commit - base_commit,
        "unequal_query_goal_flip":
            mean(unequal_query_goal_flip),
        "unequal_query_action_flip":
            mean(unequal_query_action_flip),
        "unequal_slot_swap_goal_flip":
            mean(unequal_swap_goal_flip),
        "unequal_slot_swap_action_flip":
            mean(unequal_swap_action_flip),
        "donor_slot0_target_goal":
            mean(donor0_target_goal),
        "donor_slot0_target_action":
            mean(donor0_target_action),
        "donor_slot0_nontarget_goal":
            mean(donor0_nontarget_goal),
        "donor_slot0_nontarget_action":
            mean(donor0_nontarget_action),
        "donor_slot1_target_goal":
            mean(donor1_target_goal),
        "donor_slot1_target_action":
            mean(donor1_target_action),
        "donor_slot1_nontarget_goal":
            mean(donor1_nontarget_goal),
        "donor_slot1_nontarget_action":
            mean(donor1_nontarget_action),
        "same_pair_goal_stability":
            mean(same_pair_goal_stability),
        "same_pair_action_stability":
            mean(same_pair_action_stability),
    }

    checks = {
        "attention_route":
            metrics["attention_route_accuracy"]
            >= MIN_ATTENTION_ROUTE,
        "selected_goal":
            metrics["selected_goal_accuracy"]
            >= MIN_SELECTED_GOAL,
        "autonomous_commitment":
            metrics["autonomous_commitment_correct"]
            >= MIN_COMMITMENT,
        "pair_both_queries":
            metrics["pair_both_queries_causal_chain"]
            >= MIN_PAIR_BOTH,
        "query_flip_goal":
            metrics["unequal_query_goal_flip"]
            >= MIN_QUERY_FLIP,
        "query_flip_action":
            metrics["unequal_query_action_flip"]
            >= MIN_QUERY_FLIP,
        "slot_swap_goal":
            metrics["unequal_slot_swap_goal_flip"]
            >= MIN_SWAP,
        "slot_swap_action":
            metrics["unequal_slot_swap_action_flip"]
            >= MIN_SWAP,
        "donor_slot0_target":
            min(
                metrics["donor_slot0_target_goal"],
                metrics["donor_slot0_target_action"],
            ) >= MIN_DONOR,
        "donor_slot0_nontarget":
            min(
                metrics["donor_slot0_nontarget_goal"],
                metrics["donor_slot0_nontarget_action"],
            ) >= MIN_NONTARGET_PRESERVE,
        "donor_slot1_target":
            min(
                metrics["donor_slot1_target_goal"],
                metrics["donor_slot1_target_action"],
            ) >= MIN_DONOR,
        "donor_slot1_nontarget":
            min(
                metrics["donor_slot1_nontarget_goal"],
                metrics["donor_slot1_nontarget_action"],
            ) >= MIN_NONTARGET_PRESERVE,
        "same_pair_goal_stability":
            metrics["same_pair_goal_stability"]
            >= MIN_SAME_PAIR_STABILITY,
        "same_pair_action_stability":
            metrics["same_pair_action_stability"]
            >= MIN_SAME_PAIR_STABILITY,
        "baseline_not_solving_hidden_goal":
            metrics["baseline_commitment_correct"]
            <= MAX_BASELINE_COMMITMENT,
        "commitment_advantage":
            metrics["commitment_advantage"]
            >= MIN_COMMITMENT_ADVANTAGE,
    }
    passed = all(checks.values())

    protected_after = {
        "base": sha256(BASE_CHECKPOINT),
        "memory": sha256(MEMORY_CHECKPOINT),
        "goal_decoder": sha256(GOAL_DECODER_PATH),
        "goal_options": sha256(GOAL_OPTIONS_PATH),
        "temporal_controller":
            sha256(TEMPORAL_CONTROLLER_PATH),
        "episodic_reader": sha256(EPISODIC_READER_PATH),
        "episodic_test_result":
            sha256(EPISODIC_TEST_RESULT),
    }
    if protected_before != protected_after:
        raise RuntimeError(
            "protected artifact changed during integration assay"
        )

    result = {
        "format":
            "continuous_brain.episodic_goal_skill_integration_v1",
        "integration_gate_passed": bool(passed),
        "training_performed": False,
        "new_holdout_consumed": False,
        "scope": {
            "claim":
                "late episodic selection causally controls explicit goal and temporal skill",
            "generalization_claimed": False,
            "relation_queries_motor_mapped": False,
            "planning_claimed": False,
        },
        "source": {
            "condition_family":
                "historical single-item TRAIN family only",
            "case_ids": [0, 63],
            "v023_composition_test_spent": True,
            "v0222_formal_status": "FAIL",
        },
        "provenance": provenance,
        "metrics": metrics,
        "checks": checks,
        "protected_unchanged": True,
        "v023_reader_unchanged": True,
        "v0222_components_unchanged": True,
        "details": details,
    }
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True)
    )

    print("\nINTEGRATION")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("\nGATE")
    for key, value in checks.items():
        print(
            f"{key}: {'PASS' if value else 'FAIL'}"
        )

    print("PROTECTED_UNCHANGED: true")
    print("V023_READER_UNCHANGED: true")
    print("V0222_COMPONENTS_UNCHANGED: true")
    print("NEW_HOLDOUT_CONSUMED: false")
    print(
        "EPISODIC_GOAL_SKILL_INTEGRATION_GATE:",
        "PASS" if passed else "FAIL",
    )
    print(
        "EPISODIC_GOAL_SKILL_INTEGRATION_COMPLETE"
    )
    print("result:", result_path)

    raise SystemExit(0 if passed else 4)


if __name__ == "__main__":
    main()
