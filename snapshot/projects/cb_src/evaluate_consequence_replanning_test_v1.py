#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from continuous_brain.consequence_replanning_v1 import (
    load_consequence_model,
)
from continuous_brain.episodic_goal_skill_integration_v1 import (
    EPISODIC_READER_PATH,
    GOAL_DECODER_PATH,
    decode_goal,
    select_canonical_recall,
)
from continuous_brain.multi_item_episodic_memory_v1 import (
    QUERY_FIRST,
    QUERY_SECOND,
    load_reader,
)
from continuous_brain.memory_guided_goal_option_v1 import (
    BASE_CHECKPOINT,
    MEMORY_CHECKPOINT,
    build_substrate,
    collect_decision_snapshot,
    load_goal_decoder,
    sha256,
)
from train_evaluate_consequence_replanning_v1 import (
    build_transition_dataset,
    model_metrics,
)
from evaluate_consequence_replanning_cost_repair_v1 import (
    execute_replanning,
    execute_plan_once,
)

DEV_REPAIR_RESULT = Path(
    "state/consequence_replanning_cost_repair_v1/result.json"
)
SOURCE_MODEL = Path(
    "state/consequence_replanning_v1/consequence_model.pt"
)
TEST_ROUTE_IDS = range(2304, 2560)

MIN_MODEL_NODE_ACC = .995
MIN_MODEL_BLOCK_ACC = .99
MAX_MODEL_COST_MAE = .04
MIN_MODEL_SUCCESS_ACC = .995
MIN_MODEL_VALID_ACC = .995

MIN_MEMORY_GOAL_ACC = .99
MIN_OVERALL_REPLAN_SUCCESS = .95
MIN_OPEN_SUCCESS = .95
MIN_KNOWN_SUCCESS = .95
MIN_SURPRISE_SUCCESS = .95
MAX_SURPRISE_PLAN_ONCE = .10
MIN_SURPRISE_REPLAN_TRIGGER = .95
MIN_SURPRISE_MISMATCH = .95
MIN_QUERY_PLAN_FLIP = .95
MIN_KNOWN_MODEL_OVERRIDE = .95
MIN_SURPRISE_ADVANTAGE = .80
MIN_SURPRISE_EXPOSURE = .95


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def cue_value(cue):
    return 0 if cue == "left" else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--state-dir",
        default="state/consequence_replanning_test_v1",
    )
    args = ap.parse_args()

    state_dir = Path(args.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    result_path = state_dir / "result.json"

    if not DEV_REPAIR_RESULT.is_file():
        raise FileNotFoundError(DEV_REPAIR_RESULT)
    if not SOURCE_MODEL.is_file():
        raise FileNotFoundError(SOURCE_MODEL)

    dev = json.loads(DEV_REPAIR_RESULT.read_text())
    if dev.get("development_gate_passed") is not True:
        raise RuntimeError(
            "v0.25.1 DEV repair did not pass; TEST use is not authorized"
        )
    if dev.get("split", {}).get("route_test_touched") is not False:
        raise RuntimeError("route TEST is not sealed")

    protected_paths = {
        "base": Path(BASE_CHECKPOINT),
        "memory": Path(MEMORY_CHECKPOINT),
        "goal_decoder": Path(GOAL_DECODER_PATH),
        "episodic_reader": Path(EPISODIC_READER_PATH),
        "consequence_model": SOURCE_MODEL,
        "dev_repair_result": DEV_REPAIR_RESULT,
    }
    before = {k: sha256(v) for k, v in protected_paths.items()}

    print("CONSEQUENCE_REPLANNING_TEST_V1")
    print("device: cpu")
    print("training_performed: false")
    print("route_test_ids: 2304-2559")
    print("route_test_used_exactly_once: true")
    print("test_tuning_after_run: forbidden")
    print("planner_repair_frozen: true")
    print("consequence_model_training: false")
    print("historical_planner_training: false")
    print("historical_world_model_training: false")
    print("memory_training: false")
    print("episodic_reader_training: false")
    print("goal_decoder_training: false")

    model, _ = load_consequence_model(SOURCE_MODEL, "cpu")

    # Frozen TEST transition fidelity under ordinary modeled dynamics.
    test_transition = build_transition_dataset(TEST_ROUTE_IDS)
    test_model = model_metrics(model, test_transition)

    base, cfg, admitted = build_substrate("cpu")
    decoder, _ = load_goal_decoder(GOAL_DECODER_PATH, "cpu")
    reader, _ = load_reader(EPISODIC_READER_PATH, "cpu")

    recall_cache = {}

    def recall(cid, cue):
        key = (int(cid), str(cue))
        if key not in recall_cache:
            recall_cache[key] = collect_decision_snapshot(
                base,
                cfg,
                admitted,
                int(cid),
                str(cue),
                device="cpu",
            )["recall"].clone()
        return recall_cache[key]

    memory_goal_ok = []
    query_flip = []
    known_override = []

    replanning = {
        "open": [],
        "known_block": [],
        "surprise_block": [],
    }
    plan_once = {
        "open": [],
        "known_block": [],
        "surprise_block": [],
    }

    surprise_trigger = []
    surprise_mismatch = []
    surprise_exposure = []
    details = []

    for n, cid in enumerate(TEST_ROUTE_IDS, 1):
        event1 = (cid * 17 + 11) % 1600
        event2 = (cid * 37 + 101) % 1600
        cue1 = "left" if cid % 2 == 0 else "right"
        cue2 = "right" if cue1 == "left" else "left"

        s0 = recall(event1, cue1)
        s1 = recall(event2, cue2)

        qrows = {}

        for query, expected_cue in (
            (QUERY_FIRST, cue1),
            (QUERY_SECOND, cue2),
        ):
            selected = select_canonical_recall(
                reader, s0, s1, query
            )
            goal, _ = decode_goal(
                decoder, selected["selected_recall"]
            )
            expected_goal = cue_value(expected_cue)
            memory_goal_ok.append(float(goal == expected_goal))

            r = execute_replanning(model, cid, goal)
            p = execute_plan_once(model, cid, goal)

            replanning[r["mode"]].append(float(r["success"]))
            plan_once[p["mode"]].append(float(p["success"]))

            target_approach = 0 if int(goal) == 0 else 1

            if r["mode"] == "known_block":
                known_override.append(
                    float(
                        len(r["initial_plan"]) >= 1
                        and int(r["initial_plan"][0])
                        != target_approach
                    )
                )

            if r["mode"] == "surprise_block":
                surprise_exposure.append(
                    float(
                        len(r["initial_plan"]) >= 2
                        and int(r["initial_plan"][0]) == target_approach
                        and int(r["initial_plan"][1]) == 2
                    )
                )
                surprise_trigger.append(
                    float(r["replan_count"] >= 1)
                )
                surprise_mismatch.append(
                    float(r["mismatch_count"] >= 1)
                )

            qrows[query] = {
                "goal": int(goal),
                "first_plan_action": int(r["initial_plan"][0]),
                "initial_plan": list(r["initial_plan"]),
                "mode": r["mode"],
                "replanning_success": bool(r["success"]),
                "plan_once_success": bool(p["success"]),
                "replans": int(r["replan_count"]),
                "mismatches": int(r["mismatch_count"]),
            }

        query_flip.append(
            float(
                qrows[QUERY_FIRST]["goal"]
                != qrows[QUERY_SECOND]["goal"]
                and qrows[QUERY_FIRST]["first_plan_action"]
                != qrows[QUERY_SECOND]["first_plan_action"]
            )
        )

        details.append({
            "condition_id": int(cid),
            "first": qrows[QUERY_FIRST],
            "second": qrows[QUERY_SECOND],
        })

        if n == 1 or n % 32 == 0 or n == len(TEST_ROUTE_IDS):
            print(
                f"route_test_conditions={n}/{len(TEST_ROUTE_IDS)}"
            )

    all_replan = [x for vals in replanning.values() for x in vals]
    all_once = [x for vals in plan_once.values() for x in vals]

    metrics = {
        "memory_selected_goal_accuracy": mean(memory_goal_ok),
        "replanning_success": mean(all_replan),
        "plan_once_success": mean(all_once),
        "open_replanning_success": mean(replanning["open"]),
        "known_block_replanning_success": mean(replanning["known_block"]),
        "surprise_block_replanning_success":
            mean(replanning["surprise_block"]),
        "open_plan_once_success": mean(plan_once["open"]),
        "known_block_plan_once_success": mean(plan_once["known_block"]),
        "surprise_block_plan_once_success":
            mean(plan_once["surprise_block"]),
        "surprise_initial_target_direct_rate":
            mean(surprise_exposure),
        "surprise_replan_trigger_rate":
            mean(surprise_trigger),
        "surprise_mismatch_detection_rate":
            mean(surprise_mismatch),
        "query_flip_changes_goal_and_initial_plan":
            mean(query_flip),
        "known_block_model_override_rate":
            mean(known_override),
        "surprise_replanning_advantage":
            mean(replanning["surprise_block"])
            - mean(plan_once["surprise_block"]),
    }

    checks = {
        "model_next_node":
            test_model["next_node_accuracy"] >= MIN_MODEL_NODE_ACC,
        "model_next_blocked":
            test_model["next_blocked_accuracy"] >= MIN_MODEL_BLOCK_ACC,
        "model_cost":
            test_model["cost_mae"] <= MAX_MODEL_COST_MAE,
        "model_success":
            test_model["success_accuracy"] >= MIN_MODEL_SUCCESS_ACC,
        "model_valid":
            test_model["valid_accuracy"] >= MIN_MODEL_VALID_ACC,
        "memory_selected_goal":
            metrics["memory_selected_goal_accuracy"]
            >= MIN_MEMORY_GOAL_ACC,
        "overall_replanning_success":
            metrics["replanning_success"]
            >= MIN_OVERALL_REPLAN_SUCCESS,
        "open_success":
            metrics["open_replanning_success"]
            >= MIN_OPEN_SUCCESS,
        "known_block_success":
            metrics["known_block_replanning_success"]
            >= MIN_KNOWN_SUCCESS,
        "surprise_success":
            metrics["surprise_block_replanning_success"]
            >= MIN_SURPRISE_SUCCESS,
        "surprise_exposure":
            metrics["surprise_initial_target_direct_rate"]
            >= MIN_SURPRISE_EXPOSURE,
        "surprise_plan_once_fails":
            metrics["surprise_block_plan_once_success"]
            <= MAX_SURPRISE_PLAN_ONCE,
        "surprise_replan_trigger":
            metrics["surprise_replan_trigger_rate"]
            >= MIN_SURPRISE_REPLAN_TRIGGER,
        "surprise_mismatch_detection":
            metrics["surprise_mismatch_detection_rate"]
            >= MIN_SURPRISE_MISMATCH,
        "query_flip_plan":
            metrics["query_flip_changes_goal_and_initial_plan"]
            >= MIN_QUERY_PLAN_FLIP,
        "known_block_consequence_override":
            metrics["known_block_model_override_rate"]
            >= MIN_KNOWN_MODEL_OVERRIDE,
        "surprise_replanning_advantage":
            metrics["surprise_replanning_advantage"]
            >= MIN_SURPRISE_ADVANTAGE,
    }
    passed = all(checks.values())

    after = {k: sha256(v) for k, v in protected_paths.items()}
    if before != after:
        raise RuntimeError("protected artifact changed during TEST")

    result = {
        "format":
            "continuous_brain.consequence_replanning_test_v1",
        "test_admission_passed": bool(passed),
        "training_performed": False,
        "split": {
            "route_test": [2304, 2559],
            "route_test_used": True,
            "route_test_spent": True,
            "test_tuning_after_run_forbidden": True,
        },
        "test_model_metrics": test_model,
        "test_behavior_metrics": metrics,
        "checks": checks,
        "protected_unchanged": True,
        "source_consequence_model_unchanged": True,
        "historical_planner_untouched": True,
        "details": details,
    }
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True)
    )

    print("\nTEST_MODEL")
    for k, v in test_model.items():
        print(f"{k}: {v}")

    print("\nTEST_BEHAVIOR")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    print("\nADMISSION_GATE")
    for k, v in checks.items():
        print(f"{k}: {'PASS' if v else 'FAIL'}")

    print("PROTECTED_UNCHANGED: true")
    print("SOURCE_CONSEQUENCE_MODEL_UNCHANGED: true")
    print("HISTORICAL_PLANNER_UNTOUCHED: true")
    print("ROUTE_TEST_SPENT: true")
    print("TEST_TUNING_AFTER_RUN: forbidden")
    print(
        "CONSEQUENCE_REPLANNING_TEST_ADMISSION:",
        "PASS" if passed else "FAIL",
    )
    print("CONSEQUENCE_REPLANNING_TEST_COMPLETE")
    print("result:", result_path)

    raise SystemExit(0 if passed else 4)


if __name__ == "__main__":
    main()
