#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from agent_core.actionability import PoseActionabilityMemory
from agent_core.learning import LearningPaths


def _stats_row(key, stats):
    return {
        "key": key,
        "attempts": stats.attempts,
        "successes": stats.successes,
        "failures": stats.failures,
        "recovery_attempts": stats.recoveries,
        "success_rate_pct": round(100.0 * stats.success_rate, 2),
        "last_error": stats.last_error,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect AUTHOR per-agent exact-pose and spatial interaction actionability experience.")
    parser.add_argument("--state-dir", default="state")
    parser.add_argument("--worst", type=int, default=12, help="Number of worst exact and X/Z rows to show per agent.")
    args = parser.parse_args()

    paths = LearningPaths.from_root(Path(args.state_dir)).ensure()
    output = {}
    for label in ("A", "B"):
        path = paths.agent_actionability_checkpoint(label)
        memory = PoseActionabilityMemory(path, agent_label=label)
        aggregate = defaultdict(lambda: {"attempts": 0, "successes": 0, "failures": 0, "recovery_attempts": 0})
        exact = []
        for key, stats in memory.pose_stats.items():
            parts = key.split("|")
            action = parts[0] if parts else "?"
            obj = parts[1] if len(parts) > 1 else "?"
            bucket = aggregate[(action, obj)]
            bucket["attempts"] += stats.attempts
            bucket["successes"] += stats.successes
            bucket["failures"] += stats.failures
            bucket["recovery_attempts"] += stats.recoveries
            exact.append((stats.success_rate, -stats.attempts, key, stats))

        positions = [
            (stats.success_rate, -stats.attempts, key, stats)
            for key, stats in memory.position_stats.items()
        ]
        output[label] = {
            "checkpoint": str(path),
            "pose_rows": len(memory.pose_stats),
            "position_rows": len(memory.position_stats),
            "aggregate": [
                {
                    "action": action,
                    "object": obj,
                    **row,
                    "success_rate_pct": round(100.0 * row["successes"] / max(1, row["attempts"]), 2),
                }
                for (action, obj), row in sorted(aggregate.items())
            ],
            "worst_positions": [_stats_row(key, stats) for _, _, key, stats in sorted(positions)[: max(0, int(args.worst))]],
            "worst_exact_poses": [_stats_row(key, stats) for _, _, key, stats in sorted(exact)[: max(0, int(args.worst))]],
        }

    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
