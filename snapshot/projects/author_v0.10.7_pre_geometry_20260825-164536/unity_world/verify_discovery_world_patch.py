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
          ('CreateWalkableFloor("NorthFloor"' in txt and 'CreateWalkableFloor("EastFloor"' in txt,"v0.9.4 walkable room floors configured"),
          ('go.name = "Floor";' in txt,"AI2-THOR exact Floor collider naming present"),
          ('GameObject swing = new GameObject("DoorSwing");' in txt and 'opener.MovingParts = new[] { swing };' in txt,"v0.9.4 real hinge swing configured"),
          ('PreparePortalClearance(NorthDoorCenter' in txt and 'CreateInvisibleBarrier(' in txt,"v0.9.4 physical portal clearance configured"),
          ('ClearStaticRecursive(pivot);' in txt,"runtime door static flags cleared"),
          ('RemoveStoolsFromScene(scene)' in txt and 'ValidateNoStoolsRemain(scene)' in txt,"v0.9.4 stool removal configured"),
          ('TurnaroundSize = 2.20f' in txt and 'DoorCorridorWidth = 1.25f' in txt,"v0.9.4 maneuvering envelope configured"),
          ('NorthManeuverBuffer_West' in txt and 'EastManeuverBuffer_Far' in txt,"v0.9.4 perimeter keep-out buffers configured"),
          ('ValidateRoomManeuvering' in txt,"v0.9.4 room-clearance validation configured"),
          ("UnityEngine.SceneManagement.SceneManager.MoveGameObjectToScene" in txt,"Unity SceneManager is fully qualified"),
        ]
    scene=root/"Assets/Scenes/FloorPlan1Discovery_physics.unity"
    if scene.exists(): checks.append((True,"generated discovery scene exists"))
    for ok,msg in checks: print(("OK: " if ok else "FAIL: ")+msg)
    if not all(ok for ok,_ in checks): raise SystemExit(2)
if __name__=="__main__": main()
