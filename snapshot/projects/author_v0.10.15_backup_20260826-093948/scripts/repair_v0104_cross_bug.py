#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from agent_core.migrations import repair_v0104_missing_cross_threshold


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Repair only v0.10.4 synthetic door_cross failures caused by the missing CrossThreshold helper."
    )
    parser.add_argument("--state-dir", default="state")
    args = parser.parse_args()
    result = repair_v0104_missing_cross_threshold(args.state_dir)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
