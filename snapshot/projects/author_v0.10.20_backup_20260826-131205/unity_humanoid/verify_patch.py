#!/usr/bin/env python3
from pathlib import Path
import sys

root = Path(sys.argv[1]).expanduser().resolve() if len(sys.argv) > 1 else None
if root is None:
    raise SystemExit("usage: verify_patch.py /path/to/ai2thor")

manager = root / "unity" / "Assets" / "Scripts" / "AgentManager.cs"
avatar = root / "unity" / "Assets" / "Scripts" / "HumanoidPeerAvatar.cs"
manager_text = manager.read_text(encoding="utf-8-sig") if manager.exists() else ""
avatar_text = avatar.read_text(encoding="utf-8-sig") if avatar.exists() else ""

checks = [
    (manager.exists(), "AgentManager.cs exists"),
    (avatar.exists(), "HumanoidPeerAvatar.cs exists"),
    ("V061_HUMANOID_VISUAL_REPLACEMENT" in manager_text, "v0.6.1 patch marker present"),
    ("V060_HUMANOID_SOCIAL_EMBODIMENT" not in manager_text, "v0.6.0 manager marker removed"),
    ("HumanoidPeerAvatar.ApplyAll" in manager_text, "ApplyAll hook present"),
    ("agent.TallVisCap" in avatar_text, "native TallVisCap host used"),
    ("forceRenderingOff = true" in avatar_text, "stock renderer suppression present"),
    ("shadowCastingMode = ShadowCastingMode.Off" in avatar_text, "culler-compatible shadow mode present"),
    ("v0.6.1 humanoid visual replacement applied" in avatar_text, "runtime verification marker present"),
]

for ok, label in checks:
    print(f"{'OK' if ok else 'FAIL'}: {label}")
raise SystemExit(0 if all(ok for ok, _ in checks) else 2)
