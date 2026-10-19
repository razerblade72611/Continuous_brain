# AUTHOR v0.10.4 Operator Guide

## Release purpose

v0.10.4 fixes the Level-1 door-approach failure exposed after v0.10.3 graduated Level 0.

Observed Level-1 diagnostic:

```text
150 door episodes
105 success
45 failure

A door_approach 47/75 = 62.67%
B door_approach 58/75 = 77.33%

Every one of the 45 failures:
last_action_success = True
last_action_error   = None
visible             = True
failure class       = verification-only
```

Many failed episodes ended at object-center distances between about 1.58 m and 2.21 m while the old outer verifier required <= 1.55 m. The body controller itself was targeting exact `GetInteractablePoses` poses, so this was a verifier mismatch rather than a Unity movement failure.

v0.10.4 makes pose-aware navigation authoritative end-to-end:

```text
GetInteractablePoses candidate
        ↓
physical path to exact X/Z
        ↓
align yaw + horizon
        ↓
controller checks exact pose + visibility
        ↓
ThorExecution.goal_verification
        ↓
outer Verifier consumes controller evidence
```

Object-transform-center distance remains available for diagnostics, but no longer overrides a verified interaction pose.

## Upgrade from v0.10.3 without losing A/B state

Do not reset `state/`.

```bash
cd ~/projects

mv author author_v0.10.3_backup

unzip \
  /mnt/c/Users/davei/Downloads/author_phase9_v0.10.4_pose_authoritative_navigation.zip \
  -d ~/projects

cp -a \
  ~/projects/author_v0.10.3_backup/state \
  ~/projects/author/

cd ~/projects/author
source ~/.venvs/author/bin/activate

python -m pip install -e ".[dev]"
pytest -q
```

Expected packaged regression result:

```text
156 passed
```

Verify identity:

```bash
python - <<'PY'
import agent_core
print(agent_core.__version__)
PY
```

Expected:

```text
0.10.4
```

## State compatibility

v0.10.4 preserves the existing v0.10.3 state layout:

```text
state/checkpoints/a/embodied_skills.json
state/checkpoints/b/embodied_skills.json
state/checkpoints/a/interaction_actionability.json
state/checkpoints/b/interaction_actionability.json
state/replay/embodied_replay.jsonl
state/datasets/embodied_sft.jsonl
```

The release ZIP intentionally contains no runtime `state/` directory.

## What changed in navigation verification

v0.10.3 outer verification used:

```text
visible AND object.distance <= interactionDistance + 0.30
```

With the default 1.25 m interaction distance, that created a 1.55 m object-center cutoff.

That is invalid for articulated/custom assets whose transform center may be far from the actual interaction surface.

v0.10.4 uses explicit controller evidence:

```text
pose_position_error <= max(0.16 m, 0.6 × grid_size)
yaw_error           <= half rotate step
horizon_error       <= 15 degrees
object visible       == true
lastActionSuccess    != false
```

The resulting controller record is carried in:

```text
ThorExecution.goal_verification
```

and the higher-level verifier reports:

```text
verification_basis = controller_interaction_pose
pose_goal_verified  = true/false
reason              = interaction_pose_reached / ...
object_center_distance_m = diagnostic only
```

Legacy/custom adapters without goal evidence keep the old distance fallback for compatibility, but AUTHOR's v0.10.4 adapter does not use that fallback for its own pose-aware navigation.

## Curriculum pose pinning

Levels 1–3 are deliberately staged around one exact simulator-returned interaction pose.

v0.10.3 setup could do:

```text
choose pose P
spawn near P
NavigateTo reselects P2
```

which could accidentally turn a near-approach lesson into a longer/different navigation problem.

v0.10.4 does:

```text
choose pose P
spawn according to level around P
pass P privately to body controller
navigate to P
score exact approach
```

The pinned coordinates are controller-private and are not exposed to Qwen or written into SFT conversational messages.

Pinning applies to staged Levels 1–3 for door, pickup, place, and toggle approach. Levels 4–5 remain topology-dynamic so the body can route through blocking doors and recompute accessible poses.

## Required regression experiment: Door Level 1

Do not advance to Level 2 immediately after installation.

First inspect preserved state:

```bash
python scripts/inspect_skills.py
python scripts/inspect_actionability.py --worst 30
```

Then rerun exactly one Level-1 door batch:

```bash
EPISODES=150 WORKERS=1 \
./scripts/run_training_wsl.sh \
  --curriculum door \
  --level 1
```

Then inspect:

```bash
python scripts/inspect_skills.py
python scripts/inspect_actionability.py --worst 30
python -m json.tool state/metrics/training_summary.json
```

Recommended Level-1 graduation gate:

```text
latest Level-1 episode success >= 95%
door_approach A >= 95% on the new run
door_approach B >= 95% on the new run
door_open ~= 100%
door_cross ~= 100%
controller pose-goal failures low / explainable
no systematic pinned_pose_not_available failures
```

Historical Level-1 failures from v0.10.3 remain in the cumulative checkpoint. Judge graduation from the new v0.10.4 run plus the per-level/run summary, not from the lifetime aggregate alone.

## Replay and SFT

New embodied replay rows use:

```text
author.embodied.v0104
```

New successful SFT rows use:

```text
author.embodied_sft.v0104
```

Only verified successful episodes are exported as positive SFT examples. Existing older replay/SFT rows are preserved.

## If Level 1 still fails

Inspect the new replay verification fields. A real pose-controller failure now reports a specific reason such as:

```text
interaction_pose_replans_exhausted
pinned_pose_replans_exhausted
pinned_pose_not_available
step_budget_exhausted
blocking_door_approach_failed
blocking_door_open_failed
blocking_door_cross_failed
no_interaction_pose_in_component
```

If `pose_goal_verified=true`, the approach is successful regardless of object-transform-center distance.

Do not use `--reset-memory` unless intentionally starting a new experimental lineage.
