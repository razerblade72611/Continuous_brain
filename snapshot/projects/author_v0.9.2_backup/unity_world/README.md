# AUTHOR v0.9.2 Discovery World

This patch keeps the two connected rooms introduced in v0.9.0/v0.9.1 and fixes AI2-THOR 4.3.0 reachability through both doors.

## Why v0.9.1 reported 0 reachable positions outside

AI2-THOR 4.3.0 `GetReachablePositions` flood-fills with capsule casts on layer 8 and explicitly ignores floor collisions only when the collider GameObject is named exactly `Floor`. v0.9.1 generated the new walkable cubes as `NorthFloor`, `NorthThreshold`, `EastFloor`, and `EastThreshold`, so the agent treated those floor colliders as obstacles.

v0.9.2 keeps descriptive marker parents but names the actual walkable collider children exactly `Floor`, tags/layers them as Structure, and rebuilds the scene NavMesh as before.

The existing Unity `SceneManager` compiler fix remains permanent: `UnityEngine.SceneManagement.SceneManager.MoveGameObjectToScene(...)`.

Apply and verify:

```bash
python unity_world/apply_discovery_world_patch.py ~/projects/ai2thor-humanoid
python unity_world/verify_discovery_world_patch.py ~/projects/ai2thor-humanoid
./scripts/sync_discovery_unity_to_windows.sh
```

Then rebuild the Discovery scene/player on Windows and copy the build back to WSL before running `scripts/smoke_test_discovery_world.py`.
