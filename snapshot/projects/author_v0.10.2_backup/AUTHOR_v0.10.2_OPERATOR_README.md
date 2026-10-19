# AUTHOR v0.10.2 Operator Guide

## Purpose

v0.10.2 closes the next physical-learning gap revealed by the v0.10.1 East-door probe.

`GetInteractablePoses` is extremely useful, but it is best treated as a set of **visibility/range candidate body poses**. A returned pose can still be physically bad for a particular action—for example, the humanoid capsule may block a swinging door. AUTHOR now learns that missing **actionability** layer and retries alternate body poses without asking Qwen to solve hinge clearance, grasp geometry, or receptacle access.

The normal-life boundary remains:

```text
Qwen/user chooses semantic target
        ↓
grounded objectId
        ↓
body resolves candidate interaction poses
        ↓
path cost + actionability + clearance ranking
        ↓
physical walk / yaw / camera horizon
        ↓
interaction attempt
   ┌────┴────┐
 success    failure
   │          │
 verify       ├─ record bad pose/error
   │          ├─ select alternate pose
   │          ├─ physically reposition
   │          └─ bounded retry
   ↓
public outcome returned to Qwen
```

Coordinates, exact pose keys, simulator IDs, and actionability statistics remain controller-private.

## Upgrade from v0.10.1 without losing A/B experience

Do **not** reset `state/`.

```bash
cd ~/projects

mv author author_v0.10.1_backup

unzip \
  /mnt/c/Users/davei/Downloads/author_phase9_v0.10.2_actionability_recovery.zip \
  -d ~/projects

cp -a \
  ~/projects/author_v0.10.1_backup/state \
  ~/projects/author/

cd ~/projects/author
source ~/.venvs/author/bin/activate

python -m pip install -e ".[dev]"
pytest -q
```

Expected packaged regression result:

```text
145 passed
```

Then verify package identity:

```bash
python - <<'PY'
import agent_core
print(agent_core.__version__)
PY
```

Expected:

```text
0.10.2
```

## Verify preserved staged skills

```bash
python scripts/inspect_skills.py
```

The existing v0.10.1 `door_open` Level-0 experience must still be present. `legacy_v010` must remain preserved.

The new actionability files do not exist until v0.10.2 records an interaction outcome, so this is initially valid:

```bash
python scripts/inspect_actionability.py
```

```text
pose_rows: 0
```

## Live interaction-pose preflight

The existing preflight remains useful:

```bash
GALLIUM_DRIVER=d3d12 \
MESA_D3D12_DEFAULT_ADAPTER_NAME=AMD \
python scripts/smoke_test_interaction_poses.py \
  --scene FloorPlan1Discovery_physics \
  --thor-executable \
  /home/davei/projects/ai2thor-humanoid/builds/thor-discovery-Linux64/thor-discovery-Linux64
```

Require:

```json
"ok": true
```

This verifies pose support, not guaranteed physical action success. v0.10.2's actionability memory handles that second question.

## Rerun Door Level 0 first

Before Level 1, calibrate actionability/recovery against the known doors:

```bash
EPISODES=100 WORKERS=1 \
./scripts/run_training_wsl.sh \
  --curriculum door \
  --level 0
```

Then inspect both competence and actionability:

```bash
python scripts/inspect_skills.py
python scripts/inspect_actionability.py
python -m json.tool state/metrics/training_summary.json
```

### What should change from v0.10.1

In v0.10.1, a bad direct East-door pose could produce a final failed episode. In v0.10.2:

```text
bad East-door pose
        ↓
OpenObject fails
        ↓
lastActionError captured
        ↓
exact pose actionability ↓
        ↓
alternate candidate selected
        ↓
physical reposition
        ↓
OpenObject retry
        ↓
verified success
```

The final `door_open` skill may therefore succeed while recording a recovery. `inspect_skills.py` now reports:

```text
recoveries
recovery_rate_pct
```

That is intentional. A body that notices an interaction failed and automatically adjusts position is more competent than one that requires Qwen to rediscover the same physical lesson.

## Actionability checkpoint

Each agent has independent physical-pose experience:

```text
state/checkpoints/a/interaction_actionability.json
state/checkpoints/b/interaction_actionability.json
```

A and B therefore can develop different exact-pose preferences even while sharing the same semantic model and scene.

`inspect_actionability.py` reports aggregate action/object outcomes and the worst exact pose rows:

```bash
python scripts/inspect_actionability.py --worst 20
```

A row with repeated failure should become increasingly unattractive to the motor planner. Successful recovery poses become preferred.

## Multi-worker safety

v0.10.2 does **not** let worker processes concurrently update the same actionability file.

At the beginning of a run:

```text
persistent A/B actionability
        ↓
private worker seed copies
```

At the end:

```text
worker live memory - worker seed baseline
        ↓
new delta only
        ↓
merge into persistent A/B memory
```

This prevents old actionability counts from being multiplied every time another training run or worker is used.

## Coverage metric correction

Older summaries could report values such as `coverage_pct: 320.86` because persistent cells accumulated across earlier runs were divided by the current scene's reachable-cell count.

v0.10.2 separates:

```text
run_unique_cells
persistent_unique_cells
run_coverage_pct
coverage_pct        # current-run compatibility alias
```

`run_coverage_pct` is the metric to use for curriculum decisions.

## Curriculum progression

Do not automatically advance because a run completed.

### Level 0 — direct mechanics + physical recovery

```bash
EPISODES=100 WORKERS=1 \
./scripts/run_training_wsl.sh --curriculum door --level 0
```

Desired gate per agent:

```text
door_open final success >= 95%
no single door remains systematically unrecoverable
bounded recovery rate (some recoveries are expected during learning)
```

### Level 1 — near approach

```bash
EPISODES=150 WORKERS=1 \
./scripts/run_training_wsl.sh --curriculum door --level 1
```

Suggested gate:

```text
door_approach >= 90%
door_open     >= 95%
door_cross    >= 90%
```

### Level 2 — same/local-room starts

```bash
EPISODES=200 WORKERS=1 \
./scripts/run_training_wsl.sh --curriculum door --level 2
```

Suggested gate:

```text
door_approach >= 88%
door_open     >= 95%
door_cross    >= 90%
```

### Level 3 — open multi-room

```bash
EPISODES=250 WORKERS=1 \
./scripts/run_training_wsl.sh --curriculum door --level 3
```

Suggested gate:

```text
door_approach >= 85%
door_open     >= 93%
door_cross    >= 88%
```

### Level 4 — closed-door route continuation

```bash
EPISODES=300 WORKERS=1 \
./scripts/run_training_wsl.sh --curriculum door --level 4
```

This is the first level that demonstrates the full semantic/body separation:

```text
semantic target = Apple
        ↓
Apple currently inaccessible
        ↓
body uses blocking Door as route infrastructure
        ↓
open failure? body changes pose and retries
        ↓
open → cross → recompute topology
        ↓
semantic target still = Apple
```

### Level 5 — peer interference

Only after Level 4 is healthy:

```bash
EPISODES=300 WORKERS=1 \
./scripts/run_training_wsl.sh --curriculum door --level 5
```

## Manipulation curricula

The same actionability recovery applies to:

```text
PickupObject
PutObject
ToggleObjectOn
ToggleObjectOff
OpenObject
CloseObject
```

That is important for small objects such as Knife/Spoon/Fork/Egg and for drawers/cabinets/receptacles where visibility alone may not imply a successful physical action.

After doors are healthy:

```bash
EPISODES=400 WORKERS=1 \
./scripts/run_training_wsl.sh \
  --curriculum pickup,place,toggle \
  --level 0
```

Inspect both:

```bash
python scripts/inspect_skills.py --object-details
python scripts/inspect_actionability.py --worst 30
```

## Replay and SFT behavior

Every scored interaction stage remains in:

```text
state/replay/embodied_replay.jsonl
```

v0.10.2 additionally records controller-private attempt telemetry including:

```text
last_action_success
last_action_error
interaction_recovery_used
interaction_pose_attempts
pose_key
distance_to_object
```

Only final, simulator-verified successful trajectories are exported to:

```text
state/datasets/embodied_sft.jsonl
```

A recovered successful trajectory may include the reposition/retry sequence; failed final trajectories never become positive SFT demonstrations.

## First live Escape test

After Levels 0–4 are healthy, start the normal two-agent session exactly as before and give only A:

```text
/a Escape.
```

Desired behavior:

```text
Qwen grounds selected Door
        ↓
body reaches candidate pose
        ↓
OpenObject
   │
   ├─ success → continue
   │
   └─ failure → alternate body pose → retry
        ↓
verified isOpen=True
        ↓
CrossThreshold
        ↓
verified crossed=True
        ↓
Escape verifier succeeds
```

The LLM should never need to know which exact pose failed or why.

## Do not reset

Do not use:

```bash
--reset-memory
```

unless intentionally starting a new experimental lineage. v0.10.2 is designed to continue A and B's existing histories and add actionability experience alongside them.
