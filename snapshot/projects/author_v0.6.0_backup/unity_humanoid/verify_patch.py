#!/usr/bin/env python3
from pathlib import Path
import sys

root = Path(sys.argv[1]).expanduser().resolve() if len(sys.argv) > 1 else None
if root is None:
    raise SystemExit("usage: verify_patch.py /path/to/ai2thor")
manager = root / "unity" / "Assets" / "Scripts" / "AgentManager.cs"
avatar = root / "unity" / "Assets" / "Scripts" / "HumanoidPeerAvatar.cs"
checks = [
    (manager.exists(), "AgentManager.cs exists"),
    (avatar.exists(), "HumanoidPeerAvatar.cs exists"),
    (manager.exists() and "V060_HUMANOID_SOCIAL_EMBODIMENT" in manager.read_text(encoding="utf-8-sig"), "patch marker present"),
    (manager.exists() and "HumanoidPeerAvatar.ApplyAll" in manager.read_text(encoding="utf-8-sig"), "ApplyAll hook present"),
]
for ok, label in checks:
    print(f"{'OK' if ok else 'FAIL'}: {label}")
raise SystemExit(0 if all(ok for ok, _ in checks) else 2)
