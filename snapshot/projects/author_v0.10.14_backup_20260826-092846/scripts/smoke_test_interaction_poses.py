#!/usr/bin/env python3
"""v0.10.6 live preflight for GetInteractablePoses in the actual Unity build.

This makes no LLM calls and performs no manipulation. It verifies that the custom
multi-agent build exposes current-component interaction poses with the fields the
motor layer requires before expensive curriculum training begins.
"""
from __future__ import annotations

import argparse
import json
from typing import Any

from agent_core.thor_adapter import SharedThorWorld, ThorConfig


def _row(obj: Any, poses) -> dict[str, Any]:
    example = None
    if poses:
        p = poses[0]
        example = {
            "position": {"x": round(p.x, 3), "y": round(p.y, 3), "z": round(p.z, 3)},
            "rotation": round(p.rotation, 1),
            "horizon": round(p.horizon, 1),
            "standing": bool(p.standing),
        }
    return {
        "object_type": obj.object_type,
        "name": obj.name,
        "visible_now": bool(obj.visible),
        "current_component_pose_count": len(poses),
        "example_pose": example,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify AUTHOR v0.10.6 interaction-pose support in a live AI2-THOR build.")
    parser.add_argument("--scene", default="FloorPlan1Discovery_physics")
    parser.add_argument("--thor-executable", default=None)
    parser.add_argument("--agent", type=int, default=0, choices=(0, 1))
    parser.add_argument("--max-per-category", type=int, default=8)
    args = parser.parse_args()

    cfg = ThorConfig(
        scene=args.scene,
        width=320,
        height=240,
        render_instance_segmentation=False,
        local_executable_path=args.thor_executable,
        step_delay_seconds=0.0,
        use_learned_navigation=False,
        use_interactable_poses=True,
        allow_legacy_object_center_fallback=False,
    )
    shared = SharedThorWorld(cfg, agent_count=2)
    report: dict[str, Any] = {
        "schema": "author.v0102.interaction_pose_preflight",
        "scene": args.scene,
        "agent": args.agent,
        "ok": False,
    }
    try:
        shared.start()
        adapter = shared.agent(args.agent)
        adapter.start()
        reachable = adapter.get_reachable_positions()
        world = adapter.observe()
        report["reachable_cells"] = len(reachable)

        categories = {
            "door": [o for o in world.objects.values() if str(o.object_type).lower() == "door" and o.openable],
            "pickup": [o for o in world.objects.values() if o.pickupable and not o.is_picked_up],
            "toggle": [o for o in world.objects.values() if o.toggleable],
            "receptacle": [o for o in world.objects.values() if o.receptacle],
        }
        results: dict[str, list[dict[str, Any]]] = {}
        pose_totals: dict[str, int] = {}
        for category, objects in categories.items():
            rows = []
            total = 0
            for obj in objects[: max(1, args.max_per_category)]:
                try:
                    poses = adapter.get_interactable_poses(obj.object_id, positions=reachable, standings=[True])
                    total += len(poses)
                    rows.append(_row(obj, poses))
                except Exception as exc:
                    rows.append({
                        "object_type": obj.object_type,
                        "name": obj.name,
                        "error": f"{type(exc).__name__}: {exc}",
                        "current_component_pose_count": 0,
                    })
            results[category] = rows
            pose_totals[category] = total
        report["pose_totals"] = pose_totals
        report["samples"] = results

        # A custom Discovery build should expose at least one reachable door pose;
        # manipulation categories can legitimately be absent in some current rooms.
        report["ok"] = bool(len(reachable) >= 2 and pose_totals.get("door", 0) > 0)
        if not report["ok"]:
            report["failure_hint"] = (
                "No reachable Door interaction pose was found. Do not start long training. "
                "Check custom Door metadata/colliders, current agent component, and GetInteractablePoses support first."
            )
        print(json.dumps(report, indent=2))
        return 0 if report["ok"] else 2
    finally:
        shared.close()


if __name__ == "__main__":
    raise SystemExit(main())
