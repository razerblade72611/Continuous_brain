# AUTHOR v0.9.1 Discovery World

This patch keeps `FloorPlan1_physics.unity` intact and generates a separate scene:

`Assets/Scenes/FloorPlan1Discovery_physics.unity`

The exact AI2-THOR 4.3.0 FloorPlan1 scene has two architectural door leaves. v0.9.1
turns those leaves into native `SimObjPhysics` + `CanOpen_Object` objects and attaches:

* **North Archive / Signal Room** — readable archive fragments, laptop, books, phone,
  key chain, box/statue, luminous displays.
* **East Experiment / Workshop Room** — causal-state display, workbench, lamp,
  remote, phone, book, clock, box/CD, luminous displays.

There is no teleport and no hidden reward for entering either room. The door has to be
perceived, approached, opened, and physically crossed. The new floor geometry is baked
into the Unity NavMesh.

## Apply from WSL

```bash
cd ~/projects/author
source ~/.venvs/author/bin/activate
python unity_world/apply_discovery_world_patch.py ~/projects/ai2thor-humanoid
python unity_world/verify_discovery_world_patch.py ~/projects/ai2thor-humanoid
```

Then copy the patched `SimObjType.cs` and `DiscoveryWorldBuilder.cs` into the Windows
Unity project if that project is your build source. See `scripts/build_discovery_world.ps1`.


## v0.9.1 compiler hotfix

The builder fully qualifies `UnityEngine.SceneManagement.SceneManager.MoveGameObjectToScene`. AI2-THOR also defines a class named `SceneManager`; using the unqualified name causes Unity CS0117 errors before scene generation. Re-run the patch/sync step so the Windows Unity project receives the corrected builder.
