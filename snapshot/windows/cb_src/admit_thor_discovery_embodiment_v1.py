#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def frame_sha(frame):
    if frame is None:
        return None
    try:
        return hashlib.sha256(frame.tobytes()).hexdigest()
    except Exception:
        return None


def make_controller(executable, scene, width, height):
    from ai2thor.controller import Controller

    common = dict(
        scene=scene,
        width=int(width),
        height=int(height),
        renderDepthImage=True,
    )

    attempts = [
        dict(common, local_executable_path=str(executable)),
        dict(common, executable_path=str(executable)),
    ]
    last = None
    for kwargs in attempts:
        try:
            return Controller(**kwargs)
        except TypeError as exc:
            last = exc
    raise last


def event_ok(event, label):
    md = getattr(event, "metadata", {}) or {}
    ok = bool(md.get("lastActionSuccess", True))
    if not ok:
        raise RuntimeError(
            f"{label} failed: {md.get('errorMessage', '')}"
        )
    return event


def objects(event):
    md = getattr(event, "metadata", {}) or {}
    return list(md.get("objects", []) or [])


def door_rows(event):
    rows = []
    for obj in objects(event):
        typ = str(obj.get("objectType", ""))
        if typ.lower() != "door":
            continue
        rows.append(
            {
                "objectId": str(obj.get("objectId")),
                "openable": bool(obj.get("openable", False)),
                "isOpen": bool(obj.get("isOpen", False)),
                "position": dict(obj.get("position", {}) or {}),
            }
        )
    return rows


def reachable(controller):
    ev = event_ok(
        controller.step(action="GetReachablePositions"),
        "GetReachablePositions",
    )
    pts = list(ev.metadata.get("actionReturn", []) or [])
    if not pts:
        raise RuntimeError("GetReachablePositions returned no points")
    return pts


def reachability_stats(points):
    xs = [float(p["x"]) for p in points]
    zs = [float(p["z"]) for p in points]
    return {
        "count": len(points),
        "min_x": min(xs),
        "max_x": max(xs),
        "min_z": min(zs),
        "max_z": max(zs),
        "east_extension_count": sum(x > 2.5 for x in xs),
        "north_extension_count": sum(z > 2.5 for z in zs),
    }


def find_object(event, object_id):
    for obj in objects(event):
        if str(obj.get("objectId")) == str(object_id):
            return obj
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--scene",
        default="FloorPlan1Discovery_physics",
    )
    ap.add_argument("--thor-executable", required=True)
    ap.add_argument(
        "--state-dir",
        default="state/thor_discovery_embodiment_admission_v1",
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

    protected = {
        "base":
            Path("state/child_v113_dml_smoke/infant_training.pt"),
        "memory":
            Path(
                "state/working_memory_recall_capacity_control_phase1_v1/"
                "latest.portable.pt"
            ),
        "episodic_reader":
            Path("state/multi_item_episodic_memory_v1/episodic_reader.pt"),
        "goal_decoder":
            Path("state/memory_guided_goal_option_v1/goal_decoder.pt"),
        "consequence_model":
            Path("state/consequence_replanning_v1/consequence_model.pt"),
        "consequence_test":
            Path("state/consequence_replanning_test_v1/result.json"),
    }
    for name, path in protected.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name}: {path}")

    test = json.loads(protected["consequence_test"].read_text())
    if test.get("test_admission_passed") is not True:
        raise RuntimeError(
            "v0.25.2 frozen consequence-replanning TEST did not pass"
        )
    if test.get("split", {}).get("route_test_spent") is not True:
        raise RuntimeError("v0.25.2 TEST result is not sealed/spent")

    before = {
        name: file_sha256(path)
        for name, path in protected.items()
    }

    print("THOR_DISCOVERY_EMBODIMENT_ADMISSION_V1")
    print("training_performed: false")
    print("scene:", args.scene)
    print("thor_executable:", exe)
    print("thor_executable_sha256:", file_sha256(exe))
    print("v0252_test_admission_required: true")
    print("cognitive_state_mutation_allowed: false")

    controller = None
    result = {}
    try:
        controller = make_controller(
            exe, args.scene, args.width, args.height
        )

        ev0 = controller.last_event
        md0 = ev0.metadata
        scene_name = str(md0.get("sceneName", args.scene))
        all_objects = objects(ev0)
        doors_initial = door_rows(ev0)

        if len(doors_initial) < 2:
            raise RuntimeError(
                f"expected >=2 Door objects, found {len(doors_initial)}"
            )

        initial_reach = reachable(controller)
        initial_stats = reachability_stats(initial_reach)

        frame0 = frame_sha(getattr(controller.last_event, "frame", None))
        depth0 = getattr(controller.last_event, "depth_frame", None)
        depth_present = bool(
            depth0 is not None and getattr(depth0, "size", 0) > 0
        )

        agent0 = dict(
            (controller.last_event.metadata or {})
            .get("agent", {}) or {}
        )
        yaw0 = float(
            (agent0.get("rotation", {}) or {}).get("y", 0.0)
        )

        rot = event_ok(
            controller.step(action="RotateRight", degrees=30),
            "RotateRight",
        )
        frame1 = frame_sha(getattr(rot, "frame", None))
        yaw1 = float(
            ((rot.metadata.get("agent", {}) or {})
             .get("rotation", {}) or {})
            .get("y", 0.0)
        )

        back = event_ok(
            controller.step(action="RotateLeft", degrees=30),
            "RotateLeft",
        )
        yaw2 = float(
            (((back.metadata.get("agent", {}) or {})
              .get("rotation", {}) or {})
             .get("y", 0.0))
        )

        opened = []
        for door in doors_initial:
            if not door["openable"]:
                continue
            ev = event_ok(
                controller.step(
                    action="OpenObject",
                    objectId=door["objectId"],
                    forceAction=True,
                ),
                f"OpenObject {door['objectId']}",
            )
            obj = find_object(ev, door["objectId"])
            opened.append(
                {
                    "objectId": door["objectId"],
                    "isOpen": bool(obj and obj.get("isOpen", False)),
                }
            )

        open_reach = reachable(controller)
        open_stats = reachability_stats(open_reach)
        doors_open = door_rows(controller.last_event)

        stools = [
            {
                "objectId": str(o.get("objectId")),
                "objectType": str(o.get("objectType")),
            }
            for o in objects(controller.last_event)
            if "stool" in str(o.get("objectType", "")).lower()
        ]

        checks = {
            "scene_loaded":
                bool(scene_name == args.scene),
            "two_openable_doors":
                sum(bool(d["openable"]) for d in doors_initial) >= 2,
            "door_open_state_authoritative":
                len(opened) >= 2
                and all(bool(x["isOpen"]) for x in opened),
            "reachable_positions_available":
                open_stats["count"] >= 100,
            "east_extension_reachable":
                open_stats["east_extension_count"] > 0,
            "north_extension_reachable":
                open_stats["north_extension_count"] > 0,
            "discovery_world_extent":
                open_stats["max_x"] >= 5.0
                and open_stats["max_z"] >= 5.0,
            "rgb_present":
                frame0 is not None,
            "rgb_changes_with_camera_motion":
                frame0 is not None
                and frame1 is not None
                and frame0 != frame1,
            "pose_changes_with_camera_motion":
                abs(((yaw1 - yaw0 + 180.0) % 360.0) - 180.0)
                >= 20.0,
            "pose_round_trip":
                abs(((yaw2 - yaw0 + 180.0) % 360.0) - 180.0)
                <= 1.0,
        }
        passed = all(checks.values())

        result = {
            "format":
                "continuous_brain.thor_discovery_embodiment_admission_v1",
            "admission_passed": bool(passed),
            "training_performed": False,
            "scene": scene_name,
            "thor_executable": str(exe),
            "thor_executable_sha256": file_sha256(exe),
            "initial_doors": doors_initial,
            "opened_doors": opened,
            "final_doors": doors_open,
            "initial_reachability": initial_stats,
            "open_reachability": open_stats,
            "sensor_probe": {
                "rgb_initial_sha256": frame0,
                "rgb_rotated_sha256": frame1,
                "depth_present": depth_present,
                "yaw_initial": yaw0,
                "yaw_rotated": yaw1,
                "yaw_round_trip": yaw2,
            },
            "stools_present": stools,
            "checks": checks,
            "protected_unchanged": True,
            "v0252_test_pass_preserved": True,
        }

        print("\nWORLD")
        print("scene_name:", scene_name)
        print("objects:", len(all_objects))
        print("doors:", doors_initial)
        print("stools_present:", stools)
        print("\nINITIAL_REACHABILITY")
        for k, v in initial_stats.items():
            print(f"{k}: {v}")
        print("\nOPEN_REACHABILITY")
        for k, v in open_stats.items():
            print(f"{k}: {v}")
        print("\nSENSORS")
        print("rgb_initial_sha256:", frame0)
        print("rgb_rotated_sha256:", frame1)
        print("depth_present:", depth_present)
        print("yaw_initial:", yaw0)
        print("yaw_rotated:", yaw1)
        print("yaw_round_trip:", yaw2)

        print("\nGATE")
        for k, v in checks.items():
            print(f"{k}: {'PASS' if v else 'FAIL'}")

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
        for name, path in protected.items()
    }
    if before != after:
        raise RuntimeError(
            "protected cognitive artifact changed during THOR admission"
        )

    result["protected_unchanged"] = True
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True)
    )

    print("PROTECTED_UNCHANGED: true")
    print("V0252_TEST_PASS_PRESERVED: true")
    print(
        "THOR_DISCOVERY_EMBODIMENT_ADMISSION:",
        "PASS" if result["admission_passed"] else "FAIL",
    )
    print("THOR_DISCOVERY_EMBODIMENT_ADMISSION_COMPLETE")
    print("result:", result_path)

    raise SystemExit(0 if result["admission_passed"] else 4)


if __name__ == "__main__":
    main()
