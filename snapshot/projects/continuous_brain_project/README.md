# Continuous Brain v0.5.2

> **v0.5.2 Sensor Recovery & Controller Resilience** — turns v0.5.1 sensor diagnostics into an active recovery system. Stale RGB, fully invalid trusted depth, and controller-step exceptions are quarantined as infrastructure faults; the runtime escalates from scene reset to full AI2-THOR controller recreation, actively verifies fresh RGB/depth, and only then resumes learning. The L2 architecture and 6-input/5-action neural interface remain unchanged.


Continuous Brain is an experimental persistent embodied-agent architecture combining recurrent neural state, online reinforcement learning, predictive world modeling, episodic/visual memory, intrinsic motivation, hierarchical skills, optional pretrained perception/language models, and a stable simulator/robot embodiment boundary.

v0.5.2 retains the **Spatial Cognition** milestone from v0.5.0 and the sensor/map integrity safeguards from v0.5.1. The mastered L1 navigator remains intact, but L2 replaces the privileged reachable-grid route teacher with a persistent depth-built metric map and a self-generated local waypoint.

## North star

The ultimate target is not merely an AI2-THOR agent. It is a persistent, continuously learning, multimodal embodied intelligence that combines strong pretrained foundation-model knowledge with:

- continuous recurrent state;
- autobiographical and visual memory;
- persistent spatial memory;
- online lifetime learning;
- learned action consequences;
- skill acquisition without routinely destroying old skills;
- hierarchical planning and skill composition;
- simulation-to-real embodiment behind an independent safety layer.

See `ULTIMATE_GOAL.md`.

## Architecture at v0.5

```text
pretrained visual/language/video knowledge
                  │
                  ▼
         adapters / representations
                  │
                  ▼
        Continuous recurrent brain
       ┌──────────┼──────────┐
       │          │          │
   episodic    world      hierarchical
    memory     model        skills
       │          │          │
       └──────────┼──────────┘
                  ▼
           spatial cognition
       pose belief + depth memory
       occupancy + landmarks + A*
                  │
                  ▼
        self-generated waypoint
                  │
                  ▼
       mastered L1 navigator
                  │
                  ▼
            embodiment API
```


## v0.5.2 runtime resilience

Sensor/controller failures are now outside the learning objective:

```text
sensor/controller fault
        ↓
quarantine transition (no replay / memory / learning / curriculum failure)
        ↓
scene reset + active rotate/depth verification
        ↓ if still unhealthy
stop old controller → recreate AI2-THOR controller
        ↓
reload scene + verify changed RGB and trusted depth
        ↓
resume with persistent brain/replay/spatial map
```

If all configured recovery attempts fail, the agent saves atomically and exits rather than continuing on an unverified sensor stream. `state/.../sensor_faults.jsonl` keeps bounded forensic telemetry. Restart storms are rate-limited. Controller/socket exceptions enter the same recovery path.

Default L2 launcher: `scripts/run_l2_spatial_v052.sh`.

## Neural compatibility

The core neural interface is still:

```text
6 observation values
5 high-level action intents
```

This is deliberate. v0.4.3 L0/L1 neural tensors remain shape-compatible with v0.5. Spatial cognition is a higher-level memory/planning layer rather than a forced policy-head rewrite.

The five intents remain:

```text
idle
advance
turn_left
turn_right
interact
```

## Curriculum

### L0 — orientation

Allowed actions: idle / left / right. Position is locked during the orientation episode.

### L1 — teacher-waypoint navigation

Uses short-to-long routes derived from AI2-THOR reachable positions. This is the mastered locomotion skill and is retained for rehearsal.

### L2 — self-mapped navigation

The final destination is sampled so it is a valid simulator training goal, but route topology is **not** borrowed from `GetReachablePositions`.

Instead:

```text
RGB/depth + pose belief
       ↓
persistent metric occupancy memory
       ↓
A* through free/unknown space
       ↓
self-generated local waypoint
       ↓
L1 navigation policy
```

L2 success requires real locomotion. Default minimum successful translations: 4.

## Spatial memory

`continuous_brain/spatial.py` provides a simple, inspectable metric memory designed to be replaceable later by learned SLAM components.

It currently stores:

- bounded occupancy evidence;
- free/occupied/unknown cells;
- visit counts;
- pose belief (`x`, `z`, `yaw`, confidence);
- map update counters;
- A* planner counters;
- visible semantic landmarks;
- scene identity.

Map state lives inside `brain.pt` and mastered-skill snapshots.

## Depth defenses

The custom Discovery run previously produced ~0.14 m center-depth readings during movements that AI2-THOR successfully executed. v0.5 therefore treats very short readings as untrusted map evidence by default:

```bash
--spatial-depth-min-trusted 0.25
```

The map also uses an upper-middle image band and a robust percentile instead of the old center median.

A native simulator collision is stronger evidence than depth and forcibly marks the cell ahead as occupied.

## Unknown-space behavior

A new map must be able to explore. Unknown cells are therefore not treated as walls. They are traversable with a higher A* cost:

```bash
--spatial-unknown-cost 2.5
```

Known free space becomes preferable as experience accumulates.

## Genuine L2 success

Defaults:

```text
spatial target minimum distance   1.0 m
preferred maximum distance        2.5 m
required successful translations  4
arrival tolerance                  0.18 m
```

The 0.18 m tolerance is separate from L1's 0.12 m threshold because metric map waypoints and 30° rotations can create small geometric residuals. Real translation remains mandatory, preventing the old false-success failure mode.

## Rehearsal / continual learning

L2 episodes can rehearse earlier levels:

```bash
--rehearsal-fraction 0.20
```

Replay transitions remain skill-tagged and action-mask-tagged. Persistent replay is stratified so old-skill examples survive restarts.

## Stop on mastery

Packaged v0.5 training scripts use:

```bash
--stop-on-mastery
```

When graduation is confirmed:

```text
[skill-mastered] L2 snapshot saved: .../skills/L2_mastered.pt
[stop-on-mastery] L2 graduation confirmed; saving and exiting cleanly.
```

No more accidental multi-thousand-step overtraining after mastery.

## Crash / rollback protection

State layout includes:

```text
brain.pt
brain.previous.pt
replay.pt
episodic.sqlite3
skills/
  L0_mastered.pt
  L1_mastered.pt
  L2_mastered.pt   # after spatial graduation
```

Primary checkpoints are atomic and startup falls back to `brain.previous.pt` when needed.

## Install / test

```bash
python -m pip install -e . --no-deps
python -m pytest -q
```

Expected packaged result:

```text
50 passed
```

## Recommended upgrade from your L1-mastered v0.4.3 branch

Preserve the old project, install v0.5, then copy the mastered state into a new spatial branch. Full commands are in:

```text
UPGRADE_v0.4.3_to_v0.5.0.md
```

Then:

```bash
export THOR_EXECUTABLE="$HOME/projects/ai2thor-humanoid/builds/thor-discovery-Linux64/thor-discovery-Linux64"

STATE_DIR=state/embodied_brain_v050_spatial \
  ./scripts/run_l2_spatial_v05.sh
```

## Useful scripts

- `scripts/run_l0_v05.sh` — clean L0 bootstrap with automatic stop on mastery.
- `scripts/run_l1_v05.sh` — L1 teacher navigation with automatic stop on mastery.
- `scripts/run_l2_spatial_v05.sh` — new v0.5 self-mapped L2 training.
- `scripts/verify_state.py` — read-only checkpoint/replay/skill/map verification.
- `scripts/branch_from_skill_snapshot.sh` — branch from a known-good mastered skill.

## Dashboard

`--visual` shows:

- RGB sensor;
- agent/goal/self-waypoint;
- occupancy-map overlay;
- known/free/occupied cells;
- landmarks;
- planner calls/failures;
- pose/yaw/depth;
- skill goals and graduation evidence;
- Q values and recurrent activity.

The rendered top-down map is diagnostic only. It is not directly passed to the neural policy.

## Honest limitations

v0.5 is not complete SLAM and does not claim to be:

- simulator metadata currently anchors pose belief;
- final L2 training goals are sampled from simulator-reachable destinations;
- visible object metadata currently bootstraps landmark identities/positions;
- no learned visual odometry or loop closure yet;
- no pose covariance filter yet;
- no VLM-derived semantic localization yet;
- no physical-robot safety claim.

The next spatial releases should progressively replace these privileged scaffolds rather than hide them.


## v0.5.1 sensor integrity

The dashboard now reports RGB frame sequence/hash/change, stale-camera counters, trusted-depth fraction, and sensor faults. Policy-facing depth ignores Discovery self-occlusion below the trusted threshold. Simulator training still delegates collision authority to native physics, while non-native safety mode fails closed when depth is invalid. Native collision cells are persisted as hard occupancy evidence so repeated free-space rays cannot erase a physically confirmed wall.

## v0.5.3 direction: sensor hysteresis, then cognition

v0.5.3 treats temporary invalid depth in native AI2-THOR training as unknown evidence rather than a dead controller. RGB freezes and controller exceptions remain recoverable infrastructure faults. Recovery now has hysteresis/cooldown and separate soft-reset storm protection.

This release is also an explicit project pivot: once L2 spatial cognition is experimentally usable, simulator infrastructure is frozen and development advances to semantic hierarchical skills, foundation-model grounding, temporal/world modeling, and transfer.

### Spiking / neuromorphic status

The current brain is not an SNN. It uses conventional PyTorch recurrent/dense computation because this integrates cleanly with pretrained language/vision representations and our existing continual-learning checkpoints. SNNs are now an explicit hybrid research track for fast event-driven temporal processing, sensor fusion, salience, homeostatic dynamics, and later low-level motor/reflex loops. See `NEUROMORPHIC_ROADMAP.md`.
