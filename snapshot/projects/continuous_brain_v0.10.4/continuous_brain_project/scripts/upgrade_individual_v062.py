#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from continuous_brain.motor_upgrade import attach_development_motor


def main():
    parser = argparse.ArgumentParser(description="Preserve an existing executive and attach a qualified development motor")
    parser.add_argument("source")
    parser.add_argument("destination")
    parser.add_argument("--motor-state", required=True, help="stopped motor lab with current passing qualification")
    args = parser.parse_args()
    print(json.dumps(attach_development_motor(args.source, args.destination, args.motor_state), indent=2))


if __name__ == "__main__":
    main()
