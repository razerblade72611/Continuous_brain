#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from agent_core.actionability import PoseActionabilityMemory
from agent_core.learning import LearningPaths


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect AUTHOR per-agent interaction-pose actionability experience.")
    parser.add_argument("--state-dir", default="state")
    parser.add_argument("--worst", type=int, default=12, help="Number of worst exact pose/action rows to show per agent.")
    args = parser.parse_args()

    paths = LearningPaths.from_root(Path(args.state_dir)).ensure()
    output = {}
    for label in ("A", "B"):
        path = paths.agent_actionability_checkpoint(label)
        memory = PoseActionabilityMemory(path, agent_label=label)
        aggregate = defaultdict(lambda: {"attempts": 0, "successes": 0, "failures": 0, "recoveries": 0})
        exact = []
        for key, stats in memory.pose_stats.items():
            parts = key.split("|")
            action = parts[0] if parts else "?"
            # objectId is sanitized because literal pipes are replaced with underscores.
            obj = parts[1] if len(parts) > 1 else "?"
            bucket = aggregate[(action, obj)]
            bucket["attempts"] += stats.attempts
            bucket["successes"] += stats.successes
            bucket["failures"] += stats.failures
            bucket["recoveries"] += stats.recoveries
            exact.append((stats.success_rate, -stats.attempts, key, stats))

        output[label] = {
            "checkpoint": str(path),
            "pose_rows": len(memory.pose_stats),
            "aggregate": [
                {
                    "action": action,
                    "object": obj,
                    **row,
                    "success_rate_pct": round(100.0 * row["successes"] / max(1, row["attempts"]), 2),
                }
                for (action, obj), row in sorted(aggregate.items())
            ],
            "worst_exact_poses": [
                {
                    "key": key,
                    "attempts": stats.attempts,
                    "successes": stats.successes,
                    "failures": stats.failures,
                    "recoveries": stats.recoveries,
                    "success_rate_pct": round(100.0 * stats.success_rate, 2),
                    "last_error": stats.last_error,
                }
                for _, _, key, stats in sorted(exact)[: max(0, int(args.worst))]
            ],
        }

    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
