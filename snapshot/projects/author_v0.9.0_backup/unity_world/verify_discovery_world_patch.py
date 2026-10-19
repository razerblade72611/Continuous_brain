#!/usr/bin/env python3
from pathlib import Path
import argparse

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("ai2thor_root"); args=ap.parse_args()
    root=Path(args.ai2thor_root).expanduser().resolve()/"unity"
    enum=(root/"Assets/Scripts/SimObjType.cs").read_text(encoding="utf-8-sig")
    builder=(root/"Assets/Editor/DiscoveryWorldBuilder.cs")
    checks=[
      (builder.exists(),"DiscoveryWorldBuilder.cs installed"),
      ("Door = 162" in enum,"SimObjType.Door appended"),
      ((root/"Assets/Scenes/FloorPlan1_physics.unity").exists(),"original FloorPlan1 preserved"),
    ]
    if builder.exists():
        txt=builder.read_text(encoding="utf-8-sig")
        checks += [
          ("FloorPlan1Discovery_physics.unity" in txt,"custom output scene configured"),
          ("NorthArchiveRoom" in txt,"north archive room configured"),
          ("EastExperimentRoom" in txt,"east experiment room configured"),
          ("CanOpen_Object" in txt and "SimObjSecondaryProperty.CanOpen" in txt,"native openable door configuration present"),
          ("NavMeshBuilder.BuildNavMesh" in txt,"NavMesh rebuild configured"),
        ]
    scene=root/"Assets/Scenes/FloorPlan1Discovery_physics.unity"
    if scene.exists(): checks.append((True,"generated discovery scene exists"))
    for ok,msg in checks: print(("OK: " if ok else "FAIL: ")+msg)
    if not all(ok for ok,_ in checks): raise SystemExit(2)
if __name__=="__main__": main()
