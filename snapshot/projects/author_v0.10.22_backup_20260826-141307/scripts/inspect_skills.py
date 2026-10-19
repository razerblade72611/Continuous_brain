#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent_core.embodied_skills import EmbodiedSkillLibrary
from agent_core.learning import LearningPaths


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect AUTHOR per-agent staged embodied skill checkpoints.")
    parser.add_argument("--state-dir", default="state")
    parser.add_argument("--migrate", action="store_true", help="Persist lossless legacy v0.10 checkpoint migration into the current staged schema after making a backup.")
    parser.add_argument("--object-details", action="store_true", help="Include staged by-object-type statistics.")
    args = parser.parse_args()
    paths = LearningPaths.from_root(Path(args.state_dir)).ensure()
    payload = {}
    for label in ("A", "B"):
        checkpoint = paths.agent_skill_checkpoint(label)
        lib = EmbodiedSkillLibrary(checkpoint, agent_label=label)
        if args.migrate and checkpoint.exists():
            lib.save()
        row = {
            "aggregate": lib.compact_summary(),
            "by_level": lib.level_summary(),
            "checkpoint": str(checkpoint),
            "legacy_v010_preserved": bool(lib.legacy_v010),
        }
        if args.object_details:
            row["by_object_type"] = {
                skill: {
                    obj: {
                        "attempts": stats.attempts,
                        "success_rate_pct": round(100.0 * stats.success_rate, 2),
                        "mean_steps": round(stats.mean_steps, 2),
                        "recovery_attempts": stats.recoveries,
                        "recovery_successes": stats.recovery_successes,
                        "recovery_exhausted": stats.recovery_exhausted,
                        "recovery_unclassified": stats.recovery_unclassified,
                        "recovery_success_rate_pct": round(100.0 * stats.recovery_successes / max(1, stats.recoveries), 2),
                    }
                    for obj, stats in sorted(objects.items()) if stats.attempts
                }
                for skill, objects in sorted(lib.by_object_type.items())
                if any(stats.attempts for stats in objects.values())
            }
        payload[label] = row
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
