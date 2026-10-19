#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from continuous_brain.semantic_upgrade import attach_semantic_cortex


def main():
    p = argparse.ArgumentParser(description="Attach a qualified v0.7.1 grounded-dialogue cortex to a continuation copy")
    p.add_argument("source")
    p.add_argument("destination")
    p.add_argument("--semantic-lab", required=True)
    args = p.parse_args()
    print(json.dumps(attach_semantic_cortex(args.source, args.destination, args.semantic_lab), indent=2))


if __name__ == "__main__":
    main()
