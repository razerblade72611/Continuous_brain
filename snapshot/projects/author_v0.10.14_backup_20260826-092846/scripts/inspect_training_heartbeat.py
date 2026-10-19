#!/usr/bin/env python3
"""Print the newest active AUTHOR worker heartbeat, if present."""
from __future__ import annotations

import json
from pathlib import Path


def main() -> int:
    root = Path("state/workers")
    if not root.exists():
        print("No state/workers directory found.")
        return 1
    candidates = sorted(root.glob("*/worker_*/heartbeat.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        print("No worker heartbeat found.")
        return 1
    path = candidates[0]
    print(f"heartbeat: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Could not parse heartbeat: {exc}")
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
