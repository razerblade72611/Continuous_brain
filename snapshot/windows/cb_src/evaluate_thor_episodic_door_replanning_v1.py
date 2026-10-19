#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from continuous_brain.thor_episodic_door_replanning_v1 import (
    DoorEpisode,
    bfs_path,
    choose_start_points,
    crossed_far_side,
    derive_door_geometry,
    door_rows,
    ensure_door_state,
    execute_path,
    file_sha256,
    make_controller,
    native_open_door,
    object_by_id,
    reachable,
    select_bound_episode,
    teleport,
)
from continuous_brain.memory_guided_goal_option_v1 import (
    BASE_CHECKPOINT,
    MEMORY_CHECKPOINT,
    build_substrate,
    collect_decision_snapshot,
)
from continuous_brain.multi_item_episodic_memory_v1 import (
    QUERY_FIRST,
    QUERY_SECOND,
    load_reader,
)
from continuous_brain.episodic_goal_skill_integration_v1 import (
    EPISODIC_READER_PATH,
    GOAL_DECODER_PATH,
)

# Paths are explicit so the bridge never mutates them.
CONSEQUENCE_MODEL = Path(
    "state/consequence_replanning_v1/consequence_model.pt"
)
CONSEQUENCE_TEST = Path(
    "state/consequence_replanning_test_v1/result.json"
)
THOR_ADMISSION = Path(
    "state/thor_discovery_embodiment_admission_v1/result.json"
)
V024_RESULT = Path(
    "state/episodic_goal_skill_integration_v1/result.json"
)

MODES = ("open", "known_closed", "surprise_closed")
START_COUNT = 3
MEMORY_KEY_CONDITION = 31

MIN_QUERY_TARGET = .99
MIN_SWAP_TARGET = .99
MIN_NAVIGATION = .90
MIN_OPEN_SUCCESS = .90
MIN_KNOWN_SUCCESS = .90
MIN_SURPRISE_SUCCESS = .90
MAX_SURPRISE_PLAN_ONCE = .25
MIN_SURPRISE_MISMATCH = .90
MIN_SURPRISE_REPLAN = .90
MIN_NATIVE_OPEN = .90
MIN_CROSS = .90
MIN_REPLAN_ADVANTAGE = .65


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def json_episode(ep: DoorEpisode):
    return {
        "slot": int(ep.slot),
        "label": ep.label,
        "object_id": ep.object_id,
        "position": list(ep.position),
        "outward_axis": ep.outward_axis,
        "outward_sign": int(ep.outward_sign),
        "approach": list(ep.approach),
        "far_target": list(ep.far_target),
    }


def reset_and_close(controller, scene, door_ids):
    try:
        controller.reset(scene=scene)
    except TypeError:
        controller.reset(scene)

    for object_id in door_ids:
        if not ensure_door_state(
            controller,
            object_id,
            False,
            force=True,
        ):
            raise RuntimeError(
                f"could not force-close setup door {object_id}"
            )


def current_open(controller, object_id):
    obj = object_by_id(controller, object_id)
    return bool(obj and obj.get("isOpen", False))


def make_initial_plan(controller, episode):
    is_open = current_open(controller, episode.object_id)

    pose = controller.last_event.metadata.get("agent", {}).get(
        "position", {}
    )
    start = (
        float(pose.get("x", 0.0)),
        float(pose.get("y", 0.0)),
        float(pose.get("z", 0.0)),
    )

    points = reachable(controller)
    approach_path = bfs_path(
        points,
        start,
        episode.approach,
    )

    stale_cross_path = None
    if is_open:
        # Stored while the world still matches the initial prediction.
        stale_cross_path = bfs_path(
            points,
            episode.approach,
            episode.far_target,
        )

    macros = ["navigate_approach"]
    if not is_open:
        macros.append("open_door")
    macros.append("cross_door")

    return {
        "door_open_at_plan": bool(is_open),
        "predicted_door_open_after_navigation": bool(is_open),
        "macros": macros,
        "approach_path": approach_path,
        "stale_cross_path": stale_cross_path,
    }


def run_episode(
    controller,
    *,
    scene,
    all_door_ids,
    episode,
    start_point,
    mode,
    allow_replan,
):
    reset_and_close(controller, scene, all_door_ids)
    teleport(controller, start_point, yaw=0.0)

    if mode in ("open", "surprise_closed"):
        ok = ensure_door_state(
            controller,
            episode.object_id,
            True,
            force=True,
        )
        if not ok:
            raise RuntimeError(
                f"setup could not open {episode.object_id}"
            )

    plan = make_initial_plan(controller, episode)

    nav = execute_path(
        controller,
        plan["approach_path"],
    )

    if not nav["success"]:
        return {
            "mode": mode,
            "allow_replan": bool(allow_replan),
            "success": False,
            "navigation_success": False,
            "open_macro_attempted": False,
            "native_open_success": None,
            "cross_success": False,
            "mismatch_detected": False,
            "replan_triggered": False,
            "initial_plan": plan["macros"],
            "failure": "navigation_failed",
            "navigation": nav,
        }

    if mode == "surprise_closed":
        injected = ensure_door_state(
            controller,
            episode.object_id,
            False,
            force=True,
        )
        if not injected:
            raise RuntimeError(
                f"surprise injection could not close {episode.object_id}"
            )

    actual_open = current_open(
        controller,
        episode.object_id,
    )
    predicted_open = bool(
        plan["predicted_door_open_after_navigation"]
    )
    mismatch = bool(actual_open != predicted_open)

    remaining = list(plan["macros"][1:])
    replan_triggered = False

    if mismatch and allow_replan:
        replan_triggered = True
        remaining = []
        if not actual_open:
            remaining.append("open_door")
        remaining.append("cross_door")

    open_attempted = False
    native_open = None
    cross_success = False
    cross_detail = None
    open_detail = None

    for macro in remaining:
        if macro == "open_door":
            open_attempted = True
            open_detail = native_open_door(
                controller,
                episode,
            )
            native_open = bool(
                open_detail["action_success"]
                and open_detail["is_open"]
            )
            if not native_open:
                return {
                    "mode": mode,
                    "allow_replan": bool(allow_replan),
                    "success": False,
                    "navigation_success": True,
                    "open_macro_attempted": True,
                    "native_open_success": False,
                    "cross_success": False,
                    "mismatch_detected": mismatch,
                    "replan_triggered": replan_triggered,
                    "initial_plan": plan["macros"],
                    "remaining_plan": remaining,
                    "failure": "native_open_failed",
                    "open_detail": open_detail,
                }

        elif macro == "cross_door":
            if (
                (not allow_replan)
                and mode == "surprise_closed"
                and plan["stale_cross_path"] is not None
            ):
                cross_path = plan["stale_cross_path"]
                path_source = "stale_initial_open_plan"
            else:
                points = reachable(controller)
                pose = controller.last_event.metadata.get(
                    "agent", {}
                ).get("position", {})
                current = (
                    float(pose.get("x", 0.0)),
                    float(pose.get("y", 0.0)),
                    float(pose.get("z", 0.0)),
                )
                try:
                    cross_path = bfs_path(
                        points,
                        current,
                        episode.far_target,
                    )
                except Exception as exc:
                    return {
                        "mode": mode,
                        "allow_replan": bool(allow_replan),
                        "success": False,
                        "navigation_success": True,
                        "open_macro_attempted": open_attempted,
                        "native_open_success": native_open,
                        "cross_success": False,
                        "mismatch_detected": mismatch,
                        "replan_triggered": replan_triggered,
                        "initial_plan": plan["macros"],
                        "remaining_plan": remaining,
                        "failure": "cross_path_unavailable",
                        "cross_path_error": str(exc),
                    }
                path_source = "fresh_actual_state"

            cross_detail = execute_path(
                controller,
                cross_path,
                stop_on_failure=True,
            )
            pose = cross_detail["final_pose"]
            cross_success = bool(
                cross_detail["success"]
                and crossed_far_side(episode, pose)
            )

            cross_detail = dict(cross_detail)
            cross_detail["path_source"] = path_source
            cross_detail["far_side_reached"] = bool(
                crossed_far_side(episode, pose)
            )

            if not cross_success:
                return {
                    "mode": mode,
                    "allow_replan": bool(allow_replan),
                    "success": False,
                    "navigation_success": True,
                    "open_macro_attempted": open_attempted,
                    "native_open_success": native_open,
                    "cross_success": False,
                    "mismatch_detected": mismatch,
                    "replan_triggered": replan_triggered,
                    "initial_plan": plan["macros"],
                    "remaining_plan": remaining,
                    "failure": "cross_failed",
                    "cross_detail": cross_detail,
                }

    return {
        "mode": mode,
        "allow_replan": bool(allow_replan),
        "success": bool(cross_success),
        "navigation_success": True,
        "open_macro_attempted": open_attempted,
        "native_open_success": native_open,
        "cross_success": bool(cross_success),
        "mismatch_detected": mismatch,
        "replan_triggered": replan_triggered,
        "initial_plan": plan["macros"],
        "remaining_plan": remaining,
        "open_detail": open_detail,
        "cross_detail": cross_detail,
        "failure": None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--scene",
        default="FloorPlan1Discovery_physics",
    )
    ap.add_argument("--thor-executable", required=True)
    ap.add_argument(
        "--state-dir",
        default="state/thor_episodic_door_replanning_v1",
    )
    ap.add_argument("--width", type=int, default=160)
    ap.add_argument("--height", type=int, default=120)
    args = ap.parse_args()

    exe = Path(args.thor_executable).expanduser().resolve()
    if not exe.is_file():
        raise FileNotFoundError(exe)

    state_dir = Path(args.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    result_path = state_dir / "result.json"

    protected_paths = {
        "base": Path(BASE_CHECKPOINT),
        "memory": Path(MEMORY_CHECKPOINT),
        "episodic_reader": Path(EPISODIC_READER_PATH),
        "goal_decoder": Path(GOAL_DECODER_PATH),
        "consequence_model": CONSEQUENCE_MODEL,
        "consequence_test": CONSEQUENCE_TEST,
        "thor_admission": THOR_ADMISSION,
        "v024_integration": V024_RESULT,
    }
    for name, path in protected_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name}: {path}")

    v025 = json.loads(CONSEQUENCE_TEST.read_text())
    if v025.get("test_admission_passed") is not True:
        raise RuntimeError("v0.25.2 TEST PASS is required")
    v026 = json.loads(THOR_ADMISSION.read_text())
    if v026.get("admission_passed") is not True:
        raise RuntimeError("v0.26.0 THOR admission PASS is required")
    v024 = json.loads(V024_RESULT.read_text())
    if v024.get("integration_gate_passed") is not True:
        raise RuntimeError("v0.24.0 integration PASS is required")

    before = {
        name: file_sha256(path)
        for name, path in protected_paths.items()
    }

    print("THOR_EPISODIC_DOOR_REPLANNING_V1")
    print("device: cpu")
    print("training_performed: false")
    print("scene:", args.scene)
    print("thor_executable:", exe)
    print("memory_binding: frozen_canonical_key_plus_real_door_payload")
    print("episode_payload_source: simulator_authoritative_door_metadata")
    print("raw_visual_memory_write_claimed: false")
    print("historical_planner_training: false")
    print("consequence_model_training: false")
    print("new_3d_consequence_model_training: false")
    print("motor_policy_training: false")
    print("start_count:", START_COUNT)
    print("modes:", ",".join(MODES))

    base, cfg, admitted = build_substrate("cpu")
    key0 = collect_decision_snapshot(
        base,
        cfg,
        admitted,
        MEMORY_KEY_CONDITION,
        "left",
        device="cpu",
    )["recall"]
    key1 = collect_decision_snapshot(
        base,
        cfg,
        admitted,
        MEMORY_KEY_CONDITION,
        "right",
        device="cpu",
    )["recall"]

    reader, _ = load_reader(
        EPISODIC_READER_PATH,
        "cpu",
    )

    controller = None
    rows = []
    plan_once_rows = []

    try:
        controller = make_controller(
            exe,
            args.scene,
            args.width,
            args.height,
        )

        # Build real door episode payloads from authoritative simulator metadata.
        reset_and_close(
            controller,
            args.scene,
            [d["objectId"] for d in door_rows(controller)],
        )
        closed_doors = [
            d for d in door_rows(controller)
            if d["openable"]
        ]
        if len(closed_doors) < 2:
            raise RuntimeError(
                f"need two openable doors, found {len(closed_doors)}"
            )

        closed_points = reachable(controller)

        provisional = []
        for door in closed_doors[:]:
            for d in closed_doors:
                ensure_door_state(
                    controller,
                    d["objectId"],
                    False,
                    force=True,
                )
            ensure_door_state(
                controller,
                door["objectId"],
                True,
                force=True,
            )
            open_points = reachable(controller)
            provisional.append(
                derive_door_geometry(
                    door,
                    closed_points,
                    open_points,
                    slot=0,
                )
            )

        priority = {
            "north_door": 0,
            "east_door": 1,
            "south_door": 2,
            "west_door": 3,
        }
        provisional.sort(
            key=lambda ep: (
                priority.get(ep.label, 99),
                ep.object_id,
            )
        )
        chosen = provisional[:2]
        episodes = [
            DoorEpisode(
                slot=i,
                label=ep.label,
                object_id=ep.object_id,
                position=ep.position,
                outward_axis=ep.outward_axis,
                outward_sign=ep.outward_sign,
                approach=ep.approach,
                far_target=ep.far_target,
            )
            for i, ep in enumerate(chosen)
        ]
        if len(episodes) != 2:
            raise RuntimeError("could not bind exactly two door episodes")

        all_door_ids = [d["objectId"] for d in closed_doors]

        reset_and_close(
            controller,
            args.scene,
            all_door_ids,
        )
        closed_points = reachable(controller)
        starts = choose_start_points(
            closed_points,
            episodes,
            count=START_COUNT,
        )
        if len(starts) < START_COUNT:
            raise RuntimeError(
                f"only {len(starts)} start points available"
            )

        print("\nBOUND_EPISODES")
        for ep in episodes:
            print(json.dumps(json_episode(ep), sort_keys=True))
        print("START_POINTS:", starts)

        normal_first = select_bound_episode(
            reader,
            key0,
            key1,
            episodes,
            QUERY_FIRST,
        )
        normal_second = select_bound_episode(
            reader,
            key0,
            key1,
            episodes,
            QUERY_SECOND,
        )
        swapped_first = select_bound_episode(
            reader,
            key1,
            key0,
            [episodes[1], episodes[0]],
            QUERY_FIRST,
        )
        swapped_second = select_bound_episode(
            reader,
            key1,
            key0,
            [episodes[1], episodes[0]],
            QUERY_SECOND,
        )

        query_target_checks = [
            float(normal_first["payload"].slot == 0),
            float(normal_second["payload"].slot == 1),
        ]
        swap_target_checks = [
            float(
                swapped_first["payload"].object_id
                == episodes[1].object_id
            ),
            float(
                swapped_second["payload"].object_id
                == episodes[0].object_id
            ),
        ]

        query_map = (
            (QUERY_FIRST, 0, "FIRST"),
            (QUERY_SECOND, 1, "SECOND"),
        )

        for si, start in enumerate(starts):
            for query, expected_slot, query_name in query_map:
                selected = select_bound_episode(
                    reader,
                    key0,
                    key1,
                    episodes,
                    query,
                )
                query_target_checks.append(
                    float(selected["slot"] == expected_slot)
                )
                ep = selected["payload"]

                for mode in MODES:
                    run = run_episode(
                        controller,
                        scene=args.scene,
                        all_door_ids=all_door_ids,
                        episode=ep,
                        start_point=start,
                        mode=mode,
                        allow_replan=True,
                    )
                    run.update(
                        {
                            "start_index": si,
                            "query": query_name,
                            "selected_slot": selected["slot"],
                            "selected_door": ep.object_id,
                            "selected_label": ep.label,
                            "attention": selected["attention"],
                        }
                    )
                    rows.append(run)

                    print(
                        f"episode start={si} query={query_name} "
                        f"door={ep.label} mode={mode} "
                        f"success={int(run['success'])} "
                        f"mismatch={int(run['mismatch_detected'])} "
                        f"replan={int(run['replan_triggered'])}"
                    )

                    if mode == "surprise_closed":
                        stale = run_episode(
                            controller,
                            scene=args.scene,
                            all_door_ids=all_door_ids,
                            episode=ep,
                            start_point=start,
                            mode=mode,
                            allow_replan=False,
                        )
                        stale.update(
                            {
                                "start_index": si,
                                "query": query_name,
                                "selected_slot": selected["slot"],
                                "selected_door": ep.object_id,
                                "selected_label": ep.label,
                                "attention": selected["attention"],
                            }
                        )
                        plan_once_rows.append(stale)

                        print(
                            f"plan_once start={si} query={query_name} "
                            f"door={ep.label} "
                            f"success={int(stale['success'])} "
                            f"mismatch={int(stale['mismatch_detected'])}"
                        )

    finally:
        if controller is not None:
            try:
                controller.reset(args.scene)
            except Exception:
                pass
            try:
                controller.stop()
            except Exception:
                pass

    after = {
        name: file_sha256(path)
        for name, path in protected_paths.items()
    }
    if before != after:
        raise RuntimeError(
            "protected cognitive artifact changed during 3-D bridge"
        )

    open_rows = [r for r in rows if r["mode"] == "open"]
    known_rows = [r for r in rows if r["mode"] == "known_closed"]
    surprise_rows = [
        r for r in rows
        if r["mode"] == "surprise_closed"
    ]

    native_open_rows = [
        r for r in rows
        if r["open_macro_attempted"]
    ]

    metrics = {
        "real_door_episode_count": 2,
        "start_points": len(starts),
        "query_route_target_accuracy":
            mean(query_target_checks),
        "slot_swap_target_flip_rate":
            mean(swap_target_checks),
        "navigation_success_rate":
            mean(float(r["navigation_success"]) for r in rows),
        "open_mode_success_rate":
            mean(float(r["success"]) for r in open_rows),
        "known_closed_success_rate":
            mean(float(r["success"]) for r in known_rows),
        "surprise_replanning_success_rate":
            mean(float(r["success"]) for r in surprise_rows),
        "surprise_plan_once_success_rate":
            mean(float(r["success"]) for r in plan_once_rows),
        "surprise_mismatch_detection_rate":
            mean(float(r["mismatch_detected"]) for r in surprise_rows),
        "surprise_replan_trigger_rate":
            mean(float(r["replan_triggered"]) for r in surprise_rows),
        "native_open_success_rate":
            mean(
                float(bool(r["native_open_success"]))
                for r in native_open_rows
            ),
        "cross_success_rate":
            mean(float(r["cross_success"]) for r in rows),
        "surprise_replanning_advantage":
            mean(float(r["success"]) for r in surprise_rows)
            - mean(float(r["success"]) for r in plan_once_rows),
    }

    checks = {
        "two_real_door_episodes":
            metrics["real_door_episode_count"] == 2,
        "query_controls_route_target":
            metrics["query_route_target_accuracy"]
            >= MIN_QUERY_TARGET,
        "slot_swap_controls_route_target":
            metrics["slot_swap_target_flip_rate"]
            >= MIN_SWAP_TARGET,
        "physical_navigation":
            metrics["navigation_success_rate"]
            >= MIN_NAVIGATION,
        "open_world_crossing":
            metrics["open_mode_success_rate"]
            >= MIN_OPEN_SUCCESS,
        "known_closed_planning":
            metrics["known_closed_success_rate"]
            >= MIN_KNOWN_SUCCESS,
        "surprise_replanning":
            metrics["surprise_replanning_success_rate"]
            >= MIN_SURPRISE_SUCCESS,
        "surprise_plan_once_fails":
            metrics["surprise_plan_once_success_rate"]
            <= MAX_SURPRISE_PLAN_ONCE,
        "surprise_mismatch_detected":
            metrics["surprise_mismatch_detection_rate"]
            >= MIN_SURPRISE_MISMATCH,
        "surprise_replan_triggered":
            metrics["surprise_replan_trigger_rate"]
            >= MIN_SURPRISE_REPLAN,
        "native_door_open":
            metrics["native_open_success_rate"]
            >= MIN_NATIVE_OPEN,
        "physical_crossing":
            metrics["cross_success_rate"]
            >= MIN_CROSS,
        "replanning_advantage":
            metrics["surprise_replanning_advantage"]
            >= MIN_REPLAN_ADVANTAGE,
    }
    passed = all(checks.values())

    result = {
        "format":
            "continuous_brain.thor_episodic_door_replanning_v1",
        "development_gate_passed": bool(passed),
        "training_performed": False,
        "scene": args.scene,
        "thor_executable": str(exe),
        "thor_executable_sha256": file_sha256(exe),
        "scope": {
            "claim":
                "real-3d key-value episodic selection controls physical door goal and prediction-error-triggered replanning",
            "raw_visual_memory_write_claimed": False,
            "learned_3d_consequence_model_claimed": False,
            "learned_3d_motor_skill_claimed": False,
            "general_3d_intelligence_claimed": False,
        },
        "episodes": [json_episode(ep) for ep in episodes],
        "memory_binding": {
            "key_condition_id": MEMORY_KEY_CONDITION,
            "slot0_key_semantics": "frozen admitted canonical LEFT recall",
            "slot1_key_semantics": "frozen admitted canonical RIGHT recall",
            "payload":
                "simulator-authoritative observed door identity/position",
            "late_query_reader":
                "frozen v0.23 FIRST/SECOND attention",
        },
        "start_points": [list(p) for p in starts],
        "metrics": metrics,
        "checks": checks,
        "protected_unchanged": True,
        "v0252_test_pass_preserved": True,
        "v026_admission_pass_preserved": True,
        "rows": rows,
        "surprise_plan_once_rows": plan_once_rows,
    }
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True)
    )

    print("\nDEV_METRICS")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    print("\nGATE")
    for k, v in checks.items():
        print(
            f"{k}: {'PASS' if v else 'FAIL'}"
        )

    print("PROTECTED_UNCHANGED: true")
    print("V0252_TEST_PASS_PRESERVED: true")
    print("V026_ADMISSION_PASS_PRESERVED: true")
    print(
        "THOR_EPISODIC_DOOR_REPLANNING_DEV_GATE:",
        "PASS" if passed else "FAIL",
    )
    print(
        "THOR_EPISODIC_DOOR_REPLANNING_RUN_COMPLETE"
    )
    print("result:", result_path)

    raise SystemExit(0 if passed else 4)


if __name__ == "__main__":
    main()
