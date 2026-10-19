#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from continuous_brain.dynamic_route_world_v1 import (
    ACTION_NAMES,
    MODE_NAMES,
    APPROACH_LEFT,
    APPROACH_RIGHT,
    DIRECT,
    DynamicRouteWorld,
)
from continuous_brain.consequence_replanning_v1 import (
    load_consequence_model,
    state_mismatch,
)
from continuous_brain.consequence_replanning_cost_repair_v1 import (
    plan_route_cost_consistent,
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

SOURCE_RESULT = Path("state/consequence_replanning_v1/result.json")
SOURCE_MODEL = Path("state/consequence_replanning_v1/consequence_model.pt")
DEV_ROUTE_IDS = range(2048, 2304)

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


def execute_replanning(model, condition_id, goal_side):
    env = DynamicRouteWorld(condition_id, goal_side)
    state = env.reset()

    plan = plan_route_cost_consistent(
        model, state, env.spec, goal_side
    )
    queue = list(plan["actions"])
    preds = list(plan["predictions"])
    initial_plan = list(queue)

    mismatch_count = 0
    replan_count = 0
    trace = []

    while not env.done:
        if not queue:
            plan = plan_route_cost_consistent(
                model, state, env.spec, goal_side
            )
            queue = list(plan["actions"])
            preds = list(plan["predictions"])
            replan_count += 1

        action = int(queue.pop(0))
        pred = preds.pop(0)
        actual, reward, done, info = env.step(action)
        mismatch = state_mismatch(
            pred["next_state"], actual
        )

        if mismatch:
            mismatch_count += 1
            if not done:
                plan = plan_route_cost_consistent(
                    model, actual, env.spec, goal_side
                )
                queue = list(plan["actions"])
                preds = list(plan["predictions"])
                replan_count += 1

        trace.append({
            "action": action,
            "action_name": ACTION_NAMES[action],
            "predicted_next": pred["next_state"],
            "actual_next": dict(actual),
            "mismatch": bool(mismatch),
            "reward": float(reward),
        })
        state = actual

    return {
        "success": bool(env.success),
        "mode": MODE_NAMES[env.spec.mode],
        "total_cost": float(env.total_cost),
        "steps": int(env.steps),
        "mismatch_count": int(mismatch_count),
        "replan_count": int(replan_count),
        "initial_plan": initial_plan,
        "trace": trace,
    }


def execute_plan_once(model, condition_id, goal_side):
    env = DynamicRouteWorld(condition_id, goal_side)
    state = env.reset()
    plan = plan_route_cost_consistent(
        model, state, env.spec, goal_side
    )
    actions = list(plan["actions"])

    for action in actions:
        if env.done:
            break
        env.step(action)

    return {
        "success": bool(env.success),
        "mode": MODE_NAMES[env.spec.mode],
        "total_cost": float(env.total_cost),
        "steps": int(env.steps),
        "initial_plan": actions,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--state-dir",
        default="state/consequence_replanning_cost_repair_v1",
    )
    args = ap.parse_args()

    state_dir = Path(args.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    result_path = state_dir / "result.json"

    if not SOURCE_RESULT.is_file():
        raise FileNotFoundError(SOURCE_RESULT)
    if not SOURCE_MODEL.is_file():
        raise FileNotFoundError(SOURCE_MODEL)

    source_result = json.loads(SOURCE_RESULT.read_text())
    if source_result.get("development_gate_passed") is not False:
        raise RuntimeError(
            "expected v0.25.0 formal DEV FAIL"
        )
    if (
        source_result.get("split", {})
        .get("route_test_touched") is not False
    ):
        raise RuntimeError("route TEST is not sealed")

    protected_paths = {
        "base": Path(BASE_CHECKPOINT),
        "memory": Path(MEMORY_CHECKPOINT),
        "goal_decoder": Path(GOAL_DECODER_PATH),
        "episodic_reader": Path(EPISODIC_READER_PATH),
        "source_consequence_model": SOURCE_MODEL,
        "source_result": SOURCE_RESULT,
    }
    before = {
        k: sha256(v) for k, v in protected_paths.items()
    }

    print("CONSEQUENCE_REPLANNING_COST_REPAIR_V1")
    print("device: cpu")
    print("training_performed: false")
    print("development_window_reused: 2048-2303")
    print("new_development_window_consumed: false")
    print("route_test_touched: false")
    print("repair_scope: planner_ranking_only")
    print("source_v0250_formal_status: FAIL")
    print("historical_planner_training: false")
    print("historical_world_model_training: false")
    print("consequence_model_training: false")
    print("memory_training: false")
    print("episodic_reader_training: false")
    print("goal_decoder_training: false")

    model, _ = load_consequence_model(
        SOURCE_MODEL, "cpu"
    )

    base, cfg, admitted = build_substrate("cpu")
    decoder, _ = load_goal_decoder(
        GOAL_DECODER_PATH, "cpu"
    )
    reader, _ = load_reader(
        EPISODIC_READER_PATH, "cpu"
    )

    recall_cache = {}

    def recall(cid, cue):
        key = (int(cid), str(cue))
        if key not in recall_cache:
            recall_cache[key] = (
                collect_decision_snapshot(
                    base,
                    cfg,
                    admitted,
                    int(cid),
                    str(cue),
                    device="cpu",
                )["recall"].clone()
            )
        return recall_cache[key]

    memory_goal_ok = []
    query_flip = []
    known_override = []

    replanning = {m: [] for m in MODE_NAMES}
    plan_once = {m: [] for m in MODE_NAMES}
    surprise_trigger = []
    surprise_mismatch = []
    surprise_exposure = []
    details = []

    for n, cid in enumerate(DEV_ROUTE_IDS, 1):
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
            memory_goal_ok.append(
                float(goal == expected_goal)
            )

            r = execute_replanning(
                model, cid, goal
            )
            p = execute_plan_once(
                model, cid, goal
            )
            replanning[r["mode"]].append(
                float(r["success"])
            )
            plan_once[p["mode"]].append(
                float(p["success"])
            )

            target_approach = (
                APPROACH_LEFT
                if int(goal) == 0
                else APPROACH_RIGHT
            )

            if r["mode"] == "known_block":
                known_override.append(
                    float(
                        len(r["initial_plan"]) >= 1
                        and int(r["initial_plan"][0])
                        != target_approach
                    )
                )

            if r["mode"] == "surprise_block":
                target_direct = bool(
                    len(r["initial_plan"]) >= 2
                    and int(r["initial_plan"][0])
                    == target_approach
                    and int(r["initial_plan"][1])
                    == DIRECT
                )
                surprise_exposure.append(
                    float(target_direct)
                )
                surprise_trigger.append(
                    float(r["replan_count"] >= 1)
                )
                surprise_mismatch.append(
                    float(r["mismatch_count"] >= 1)
                )

            qrows[query] = {
                "goal": int(goal),
                "first_plan_action":
                    int(r["initial_plan"][0]),
                "initial_plan":
                    list(r["initial_plan"]),
                "mode": r["mode"],
                "replanning_success":
                    bool(r["success"]),
                "plan_once_success":
                    bool(p["success"]),
                "replans": int(r["replan_count"]),
                "mismatches":
                    int(r["mismatch_count"]),
            }

        query_flip.append(
            float(
                qrows[QUERY_FIRST]["goal"]
                != qrows[QUERY_SECOND]["goal"]
                and
                qrows[QUERY_FIRST]["first_plan_action"]
                != qrows[QUERY_SECOND]["first_plan_action"]
            )
        )

        details.append({
            "condition_id": int(cid),
            "first": qrows[QUERY_FIRST],
            "second": qrows[QUERY_SECOND],
        })

        if (
            n == 1
            or n % 32 == 0
            or n == len(DEV_ROUTE_IDS)
        ):
            print(
                f"repair_dev_conditions="
                f"{n}/{len(DEV_ROUTE_IDS)}"
            )

    all_replan = [
        x for vals in replanning.values() for x in vals
    ]
    all_once = [
        x for vals in plan_once.values() for x in vals
    ]

    metrics = {
        "memory_selected_goal_accuracy":
            mean(memory_goal_ok),
        "replanning_success":
            mean(all_replan),
        "plan_once_success":
            mean(all_once),
        "open_replanning_success":
            mean(replanning["open"]),
        "known_block_replanning_success":
            mean(replanning["known_block"]),
        "surprise_block_replanning_success":
            mean(replanning["surprise_block"]),
        "open_plan_once_success":
            mean(plan_once["open"]),
        "known_block_plan_once_success":
            mean(plan_once["known_block"]),
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
            metrics[
                "query_flip_changes_goal_and_initial_plan"
            ] >= MIN_QUERY_PLAN_FLIP,
        "known_block_consequence_override":
            metrics["known_block_model_override_rate"]
            >= MIN_KNOWN_MODEL_OVERRIDE,
        "surprise_replanning_advantage":
            metrics["surprise_replanning_advantage"]
            >= MIN_SURPRISE_ADVANTAGE,
    }
    passed = all(checks.values())

    after = {
        k: sha256(v) for k, v in protected_paths.items()
    }
    if before != after:
        raise RuntimeError(
            "protected artifact changed during repair evaluation"
        )

    result = {
        "format":
            "continuous_brain.consequence_replanning_cost_repair_v1",
        "development_gate_passed":
            bool(passed),
        "source_v0250_formal_status":
            "FAIL",
        "repair_classification":
            "evaluation/planner-ranking construction bug",
        "training_performed":
            False,
        "split": {
            "development_window_reused":
                [2048, 2303],
            "new_development_window_consumed":
                False,
            "route_test":
                [2304, 2559],
            "route_test_touched":
                False,
        },
        "metrics": metrics,
        "checks": checks,
        "protected_unchanged": True,
        "source_consequence_model_unchanged":
            True,
        "historical_planner_untouched":
            True,
        "details": details,
    }
    result_path.write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
    )

    print("\nREPAIRED_DEV_BEHAVIOR")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    print("\nGATE")
    for k, v in checks.items():
        print(
            f"{k}: {'PASS' if v else 'FAIL'}"
        )

    print("PROTECTED_UNCHANGED: true")
    print("SOURCE_CONSEQUENCE_MODEL_UNCHANGED: true")
    print("HISTORICAL_PLANNER_UNTOUCHED: true")
    print("ROUTE_TEST_TOUCHED: false")
    print(
        "CONSEQUENCE_REPLANNING_COST_REPAIR_DEV_GATE:",
        "PASS" if passed else "FAIL",
    )
    print(
        "CONSEQUENCE_REPLANNING_COST_REPAIR_COMPLETE"
    )
    print("result:", result_path)

    raise SystemExit(0 if passed else 4)


if __name__ == "__main__":
    main()
