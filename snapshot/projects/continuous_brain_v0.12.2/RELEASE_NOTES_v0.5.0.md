# Continuous Brain v0.5.0 — Spatial Cognition

v0.5.0 is the first release where the embodied stack begins to **generate its own navigation waypoint from persistent sensory memory** instead of relying on an AI2-THOR reachable-grid route teacher.

It preserves the v0.4.x 6-input / 5-intent neural interface, so L0/L1 mastered neural weights remain shape-compatible.

## New L2: self-mapped point navigation

L0 and L1 remain unchanged:

- **L0** — orient toward a goal direction.
- **L1** — teacher-waypoint navigation over the simulator reachable grid.

v0.5 changes L2 into a spatial-cognition bootstrap:

- the simulator samples a physically attainable final training destination;
- the L2 route is **not** generated from `GetReachablePositions` topology;
- RGB/depth plus pose-belief updates build a persistent metric occupancy map;
- an internal A* planner treats known occupied cells as blocked, known free cells as cheap, and unknown cells as traversable but costly;
- the planner generates the active waypoint consumed by the same recurrent navigation policy that mastered L1;
- blocked native `MoveAhead` attempts write strong obstacle evidence directly into the map and trigger replanning.

This is an intentional bridge from privileged simulator navigation to later fully learned spatial cognition.

## Persistent spatial memory

New `SpatialMemory2D` state includes:

- metric occupancy evidence;
- visit counts;
- scene-bounded persistent map geometry;
- a 2-D pose-belief interface with confidence;
- depth integration counters;
- planner success/failure counters;
- visible semantic landmark memory (currently metadata-assisted bootstrap labels);
- map/landmark persistence inside normal and mastered-skill checkpoints.

The spatial map is scene-guarded. A map saved for one scene is not silently loaded into another.

## Defensive depth handling

The real Discovery run previously showed center-depth values around `0.14 m` while `MoveAhead` was actually safe. v0.5 therefore does not treat sub-threshold readings as obstacle endpoints for mapping by default.

Depth mapping uses:

- an upper-middle image band;
- a robust percentile rather than the old center median;
- `--spatial-depth-min-trusted` (default `0.25 m`) to reject likely floor/body/self-occlusion;
- bounded log-odds evidence;
- native-physics collision evidence that overrides stale free-space estimates.

This is a simulator-learning defense, **not** a physical-robot safety certification.

## Self-generated waypoint planning

The planner is intentionally simple and auditable:

```text
pose belief + depth map + final goal
              ↓
      persistent occupancy grid
              ↓
       A* over free/unknown space
              ↓
      self-generated local waypoint
              ↓
       mastered L1 navigation policy
```

Unknown cells remain traversable so a new map can explore. They carry a configurable higher planning cost (`--spatial-unknown-cost`, default `2.5`).

## Genuine L2 success

L2 cannot succeed simply because it started close to the goal. Defaults require:

- final metric distance <= `0.18 m`;
- at least **4 successful physical translations** during the episode.

The slightly larger L2 arrival tolerance accounts for 30° turn quantization and self-generated metric waypoints while still requiring real locomotion.

## Stop on mastery

New:

```bash
--stop-on-mastery
```

When the configured curriculum level satisfies graduation:

1. the mastered skill snapshot is written atomically;
2. replay/checkpoint state is saved;
3. the run exits cleanly.

The packaged v0.5 training scripts enable this by default so mastered levels no longer continue for thousands of unnecessary steps.

## Dashboard

The browser dashboard now overlays the persistent occupancy map on the top-down diagnostic view and displays:

- known/free/occupied cell counts;
- remembered landmark count;
- planner calls/failures;
- waypoint source (`spatial-map` vs `teacher-reachable-grid`);
- pose/map persistence diagnostics.

The top-down dashboard is still diagnostic; the neural policy receives the six configured observation slots plus its visual embedding, not the rendered dashboard.

## Persistence / compatibility defenses

v0.5 keeps all v0.4.3 defenses:

- atomic primary checkpoint writes;
- `brain.previous.pt` fallback;
- persistent stratified replay;
- skill-tagged replay masks;
- mastered-skill snapshots;
- per-level statistics.

Additional spatial defenses:

- map format versioning;
- map size cap;
- scene mismatch rejection;
- bounded/quantized dashboard payload;
- compact JSON status instead of serializing raw map arrays;
- map-state migration that safely starts empty when upgrading a v0.4.3 state.

## Validation summary

Packaged source validation:

- **50 pytest tests passed**;
- Python compile check passed;
- dashboard JavaScript syntax check passed;
- deterministic L2 spatial oracle: **20/20 successes**;
- neural hierarchical smoke: L1-trained recurrent policy resumed into L2 and reached **97 genuine L2 goals** during the 2,000-tick spatial phase, with no blocked moves or planner failures in the unobstructed fake world;
- generated v0.4.3 checkpoint resumed under v0.5.0 and migrated to checkpoint format 7.

## Important limitations

v0.5 is a spatial-cognition **bootstrap**, not complete SLAM:

- pose belief is currently anchored by simulator pose metadata;
- final L2 training goals are sampled from known-reachable simulator positions, although the route is not;
- semantic landmarks currently use visible simulator object metadata as labels/positions;
- there is no learned visual odometry, loop closure, pose covariance filter, or VLM-derived object localization yet;
- the build container cannot execute the user's custom `FloorPlan1Discovery_physics` Unity binary, so the new L2 path must still be validated on that machine.

Those limitations are deliberate seams for later v0.5.x/v0.6 work rather than hidden dependencies.
