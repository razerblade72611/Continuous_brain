#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from agent_core.migrations import repair_v01019_place_transport_regression


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove only v0.10.19 Level-4 place transport-regression failures from persistent skill counters."
    )
    parser.add_argument("--state-dir", default="state")
    args = parser.parse_args()
    print(json.dumps(repair_v01019_place_transport_regression(args.state_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
