#!/usr/bin/env python3
"""Install AUTHOR v0.9.4 Discovery World source into an AI2-THOR 4.3 checkout.

This does not mutate FloorPlan1_physics.unity directly. It installs the editor
builder plus the appended Door SimObjType. Unity then generates a separate
FloorPlan1Discovery_physics.unity scene.
"""
from __future__ import annotations
from pathlib import Path
import argparse, shutil, re

MARKER = "AUTHOR v0.9.0 discovery-world types"

def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument("ai2thor_root")
    args=ap.parse_args()
    root=Path(args.ai2thor_root).expanduser().resolve()
    unity=root/"unity"
    enum_path=unity/"Assets/Scripts/SimObjType.cs"
    editor_dir=unity/"Assets/Editor"
    scripts_dir=unity/"Assets/Scripts"
    scene=unity/"Assets/Scenes/FloorPlan1_physics.unity"
    if not enum_path.exists() or not scene.exists():
        raise SystemExit(f"Not an expected AI2-THOR checkout: {root}")
    editor_dir.mkdir(parents=True, exist_ok=True)
    scripts_dir.mkdir(parents=True, exist_ok=True)

    source=Path(__file__).with_name("DiscoveryWorldBuilder.cs")
    target=editor_dir/"DiscoveryWorldBuilder.cs"
    shutil.copy2(source,target)
    sanitizer_source=Path(__file__).with_name("DiscoveryRuntimeSanitizer.cs")
    sanitizer_target=scripts_dir/"DiscoveryRuntimeSanitizer.cs"
    shutil.copy2(sanitizer_source,sanitizer_target)

    text=enum_path.read_text(encoding="utf-8-sig")
    if "Door = 162" not in text:
        needle="    ScreenSheet = 161,\n}"
        replacement=("    ScreenSheet = 161,\n"
                     "    // AUTHOR v0.9.0 discovery-world types\n"
                     "    Door = 162,\n}")
        if needle not in text:
            raise SystemExit("Could not find expected ScreenSheet=161 enum tail in SimObjType.cs")
        backup=enum_path.with_suffix(enum_path.suffix+".v090.bak")
        if not backup.exists(): shutil.copy2(enum_path,backup)
        text=text.replace(needle,replacement,1)
        enum_path.write_text(text,encoding="utf-8")
    print(f"Installed: {target}")
    print(f"Installed: {sanitizer_target}")
    print(f"Door SimObjType: {'OK' if 'Door = 162' in enum_path.read_text(encoding='utf-8-sig') else 'MISSING'}")
    print("Next: run Unity -executeMethod DiscoveryWorldBuilder.BuildFloorPlan1Discovery, then build FloorPlan1Discovery_physics.")
    return 0

if __name__ == "__main__": raise SystemExit(main())
