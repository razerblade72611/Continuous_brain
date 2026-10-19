# Changelog

## 0.10.16

- Replace Discovery-prop `PlaceObjectAtPoint` reset with exact `SetObjectPoses(..., placeStationary=True)` restoration.
- Capture and verify full position + rotation, including in-place tipped props.
- Preserve every current moveable/pickupable object in the exact-state request.
- Clear held objects before independent episode setup so successful pickup lessons cannot poison the next lesson.
- Bump replay/SFT schemas to `author.embodied.v01016` / `author.embodied_sft.v01016`.
- Preserve route-safe peer parking, dynamic blocker replanning, transport recovery, and component-verified Level-4 semantic routes.

## 0.10.15
- Restore displaced generated `Discovery_*` props before curriculum setup.
- Detect vertical as well as planar prop displacement.
- Park passive Level-4 peer away from the verified blocking-door approach corridor.
- Add bounded dynamic-blocker replanning to exact interaction-pose navigation.
- Bump embodied replay schema to `author.embodied.v01015` and SFT schema to `author.embodied_sft.v01015`.
- Add `run_id` and `worker` to embodied replay rows.
- Add `discovery_prop_restores` metric.
- Preserve v0.10.14 component-verified semantic target selection and v0.10.12 transport recovery.
