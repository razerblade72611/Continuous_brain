#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from agent_core.migrations import repair_v01022_place_destination_regression

parser = argparse.ArgumentParser(description="Remove replay-proven v0.10.22 Level-4 place destination-contract regression labels from skill counters without altering replay.")
parser.add_argument("--state-dir", default="state")
args = parser.parse_args()
print(json.dumps(repair_v01022_place_destination_regression(args.state_dir), indent=2, sort_keys=True))
