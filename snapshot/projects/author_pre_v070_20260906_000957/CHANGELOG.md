# Changelog

## 0.10.26

- Add deterministic private-motor handoff for explicit Discovery `ExitRegion` tasks.
- Preserve an LLM-grounded Door choice when available; otherwise privately select the nearest physical Door without exposing IDs or coordinates to the semantic planner.
- Stage exit execution through learned `door_approach`, `door_open`, and `door_cross` skills.
- Override raw preparatory LLM movement during an explicit exit task so local vision-language drift cannot bypass the trained doorway controller.
- Keep stock/non-Discovery scenes unchanged because they lack the connected-region verifier.
- Retain all v0.10.25 training fixes and v0.10.24 broken-toggle affordance guards.
- Replay/SFT schemas advance to `author.embodied.v01026` / `author.embodied_sft.v01026` for any future training under this release.

## 0.10.25
- Fix Level-4 toggle setup-search starvation caused by passive-agent pose persistence across rejected setups.
- Park the passive agent in a neutral maneuverable cell before blocked-route qualification, then retain route-specific peer parking after tuple selection.
- Preserve the v0.10.24 broken-target admission/runtime affordance guard and all previously qualified pickup/place/door behavior.
- Bump replay/SFT schemas to `author.embodied.v01025` / `author.embodied_sft.v01025`.
- Add v0.10.25 regression tests; full suite: 261 passed.

## 0.10.24
- Exclude AI2-THOR toggle targets whose private raw metadata reports `isBroken=True`; broken Laptops/CellPhones remain object-type toggleable but are physically invalid ToggleObjectOn lessons.
- Preserve healthy Laptop participation and all existing Level-4 semantic-route behavior.
- Add replay-safe/idempotent repair for the four v0.10.23 broken-Laptop toggle failures without removing successful toggle-approach evidence.
- Add v0.10.24 regression tests; full suite: 254 passed.


## 0.10.23

- Added trainer-private physical post-cross destination-leg preflight for Level-4 place while carrying an object.
- Invalid source/pose destination legs are skipped before scoring rather than taught as motor failures.
- Added trusted single-cell pinned-terminal injection after the semantic target component is already visible, preventing synthetic `pinned_pose_not_available` failures without bypassing closed doors.
- Added `repair_v01022_place_destination_regression.py` for the two replay-proven v0.10.22 destination-contract labels; raw replay and successful evidence are preserved.
- Added v0.10.23 regression tests; full suite: 249 passed.

## 0.10.22

- Replaced the v0.10.21 body-only arrival-yaw door admission test with the actual orientation-aware carried planner.
- Added held-object AABB-derived stand-off for opening closed doors.
- Added pre-score Level-4 carried-route admission so motor-incompatible setups are `skip_not_failure` rather than synthetic negative labels.
- Dynamic blocker replanning now blacklists the cell attempted by the failed carried primitive instead of inferring from a body-only path.
- CrossThreshold uses the carried planner to reach the portal staging pose already aligned for straight crossing.
- Added explicit execution-exception reason telemetry.
- Added replay-driven, idempotent v0.10.21 place route-admission regression repair.
- Added v0.10.22 regression tests; full suite: 244 passed.


## 0.10.21

- Preserve the stable v0.10.20 final place/release contract; fix only remaining carried-object route mechanics.
- Add orientation-aware carried routing to exact interaction poses, with body rotations allowed only in locally clearance-qualified cells.
- Permit forward/back/strafe translation so a held object can keep a stable body yaw through tight corridors and arrive at the pinned pose already aligned.
- Require carried-object door interaction poses to be laterally centered in the actual opening and remove unsafe jamb-side generic fallback.
- Recognize Unity blocker errors using the real `blocking the Agent` wording and replan around the rejected movement cell.
- Reuse a physically preflighted pinned place pose after topology changes when its cell remains reachable, while retaining authoritative final pose verification.
- Fail closed when no carried-clearance route is available rather than falling back to the body-only navigator.
- Add `repair_v01020_place_carry_route_regression.py`; it backs up A/B checkpoints, preserves raw replay, and removes only replay-proven v0.10.20 carried-route implementation failures.
- Bump replay/SFT schemas to `author.embodied.v01021` / `author.embodied_sft.v01021`.
- Trainer id: `carried_orientation_clearance_curriculum`.

## 0.10.20

- Fix Level-4 place carried-object doorway transport without changing the now-stable placement/release contract.
- Add safe source-side pre-alignment 0.85–1.25 m from the jamb, scaled by held-object AABB footprint.
- Cross the portal on one centerline grid lane with MoveAhead-only traversal and no body rotations inside the portal clearance zone.
- Suppress the generic zero-vector final-facing turn for CrossThreshold staging via `path_to_actions(..., face_target=False)`.
- Prefer carry-safe door interaction poses that can be entered already facing the door.
- Add `repair_v01019_place_transport_regression.py`, which backs up A/B checkpoints and removes only replay-proven v0.10.19 held-item/PortalBoundary transport failures while preserving raw replay.
- Bump embodied/SFT schemas to `author.embodied.v01020` / `author.embodied_sft.v01020`.
- Trainer id: `carried_object_clearance_curriculum`.


## 0.10.19

- Use the closed Door AABB center as the physical portal center instead of the hinge/pivot `position`.
- Stage CrossThreshold on the closed-door source side near the portal centerline and cross to a deeper far-side cell.
- Stop treating arbitrary cardinal adjacency in `GetReachablePositions` as proof of a traversable NavMesh edge.
- Promote provisional door-cross misses when subsequent embodied motion authoritatively enters the newly connected component.
- Preflight `PutObject(forceAction=False)` from the exact Level-4 placement interaction pose and pin scored place navigation to that verified pose.
- Preserve canonical scene reset, route-safe peer parking, receptacle-open preconditions, transport recovery, and persistent A/B state.
- Add an idempotent v0.10.18 Level-4 place regression repair that backs up checkpoints and preserves raw replay.
- Bump replay/SFT schemas to `author.embodied.v01019` / `author.embodied_sft.v01019`.

## 0.10.18

- Verify routed door crossing by closed-component -> newly connected component transition.
- Choose CrossThreshold target on the actual newly opened NavMesh component boundary.
- Add scored `place_receptacle_open` precondition for Drawer/Cabinet-style targets.
- Preflight and cache physical source/receptacle PutObject feasibility before scoring.
- Remove remaining place-path dependence on approximate surface-point cleanup.
- Bump replay/SFT schemas to `author.embodied.v01018` / `author.embodied_sft.v01018`.

## 0.10.17
- Capture canonical position + rotation for every movable/pickupable scene object, not only `Discovery_*` props.
- Exact-reset displaced/missing movable objects with `SetObjectPoses(placeStationary=True)` before each independent lesson.
- Remove the post-pickup `PlaceObjectAtPoint` cleanup that could corrupt stock target support/rotation state.
- Emit setup-search progress every 25 unscored setup rejections so a long Level-4 search is observable.
- Replay/SFT schemas bumped to `author.embodied.v01017` / `author.embodied_sft.v01017`.


## 0.10.17

- Replace Discovery-prop `PlaceObjectAtPoint` reset with exact `SetObjectPoses(..., placeStationary=True)` restoration.
- Capture and verify full position + rotation, including in-place tipped props.
- Preserve every current moveable/pickupable object in the exact-state request.
- Clear held objects before independent episode setup so successful pickup lessons cannot poison the next lesson.
- Bump replay/SFT schemas to `author.embodied.v01017` / `author.embodied_sft.v01017`.
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
