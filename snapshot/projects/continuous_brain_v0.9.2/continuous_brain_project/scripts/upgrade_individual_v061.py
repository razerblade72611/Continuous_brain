#!/usr/bin/env python3
"""Preserve a stopped individual in a new directory, optionally attach a frozen motor."""
import argparse
import json
import sys
from pathlib import Path

# Run the bundled source, even if an older version is installed in the venv.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from continuous_brain.upgrade import upgrade_individual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("destination")
    parser.add_argument("--legacy-motor", help="trusted, stopped legacy THOR state directory (optional)")
    args = parser.parse_args()
    print(json.dumps(upgrade_individual(args.source, args.destination, args.legacy_motor), indent=2))


if __name__ == "__main__":
    main()
