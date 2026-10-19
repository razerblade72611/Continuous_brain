#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from agent_core.migrations import repair_v01018_place_regression


def main() -> int:
    parser = argparse.ArgumentParser(description="Unscore known AUTHOR v0.10.18 Level-4 place trainer/verifier regressions while preserving raw replay.")
    parser.add_argument("--state-dir", default="state")
    args = parser.parse_args()
    print(json.dumps(repair_v01018_place_regression(args.state_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
