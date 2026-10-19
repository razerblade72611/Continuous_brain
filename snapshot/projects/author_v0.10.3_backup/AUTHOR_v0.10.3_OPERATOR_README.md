# AUTHOR v0.10.3 Operator Guide

## Release purpose

v0.10.3 fixes the recovery failure exposed by the v0.10.2 EastExperiment-door run.

v0.10.2 learned exact `(x,z,yaw,horizon)` actionability, but a moving door can have an entire bad physical standing region. A failed X/Z cell could therefore consume all three retries by trying different yaw/horizon variants from essentially the same collision geometry.

v0.10.3 adds **spatial actionability generalization + reliable recovery** while preserving semantic/body separation:

```text
Qwen selects semantic target
        ↓
body chooses candidate pose
        ↓
interaction fails
        ↓
record exact pose + X/Z region
        ↓
exclude failed moving-object region
        ↓
prefer safe-clearance alternatives
        ↓
reposition physically
        ↓
retry SAME semantic action
        ↓
verify simulator state
```

Coordinates, object IDs, actionability rows, recovery selection, and simulator errors remain controller-private.

## Upgrade from v0.10.2 without losing A/B experience

Do not reset `state/`.

```bash
cd ~/projects

mv author author_v0.10.2_backup

unzip \
  /mnt/c/Users/davei/Downloads/author_phase9_v0.10.3_spatial_actionability_recovery.zip \
  -d ~/projects

cp -a \
  ~/projects/author_v0.10.2_backup/state \
  ~/projects/author/

cd ~/projects/author
source ~/.venvs/author/bin/activate

python -m pip install -e ".[dev]"
pytest -q
```

Expected packaged regression result:

```text
150 passed
```

Verify package identity:

```bash
python - <<'PY'
import agent_core
print(agent_core.__version__)
PY
```

Expected:

```text
0.10.3
```

## State migration behavior

Existing v0.10.2 actionability files are reused automatically:

```text
state/checkpoints/a/interaction_actionability.json
state/checkpoints/b/interaction_actionability.json
```

v0.10.3 reconstructs X/Z `position_stats` from every existing exact-pose row the first time the file is loaded, so A and B immediately benefit from their prior East-door experience.

Existing staged skill checkpoints are also preserved. v0.10.2's old `recoveries` count is migrated as `recovery_attempts`. Because old checkpoints did not distinguish successful recovery from exhausted recovery, those historical attempts appear as `recovery_unclassified`. New v0.10.3 attempts are classified explicitly.

## New recovery metrics

`python scripts/inspect_skills.py` now reports:

```text
recovery_attempts
recovery_successes
recovery_exhausted
recovery_unclassified
recovery_attempt_rate_pct
recovery_success_rate_pct
```

Interpretation:

```text
recovery_attempts     = semantic interaction needed internal pose recovery
recovery_successes    = recovery ultimately completed the same semantic action
recovery_exhausted    = bounded recovery budget ended without success
recovery_unclassified = historical v0.10.2 recovery attempts lacking outcome labels
```

## Spatial actionability inspection

Run:

```bash
python scripts/inspect_actionability.py --worst 30
```

v0.10.3 reports both:

```text
pose_rows
position_rows
worst_exact_poses
worst_positions
```

A bad East-door X/Z region should now become unattractive even for yaw/horizon combinations never individually attempted.

## Recovery policy

Normal approach may trade actionability against route cost.

Recovery is deliberately different:

1. Exclude exact failed pose.
2. For `OpenObject` / `CloseObject`, exclude the whole failed X/Z cell.
3. Prefer candidates outside `interaction_recovery_exclusion_radius_m`.
4. If available, restrict moving-object recovery to candidates at or beyond `preferred_open_clearance_m`.
5. Rank actionability/clearance much more strongly than movement cost.
6. Retry the same semantic action up to the configured bounded limit.

Defaults in `config.yaml`:

```yaml
navigation:
  max_interaction_action_retries: 3
  preferred_open_clearance_m: 1.0
  interaction_recovery_exclusion_radius_m: 0.50
  interaction_recovery_min_position_change_m: 0.25
```

The strict radius can relax in a narrow scene if no valid candidate exists, but v0.10.3 never deliberately retries the same failed X/Z cell for a moving-object interaction.

## Important Level-4 fix

Closed doors used internally to continue toward another semantic target now use the same actionability/recovery subsystem.

Example:

```text
semantic target = Apple
        ↓
Apple inaccessible
        ↓
body reaches blocking Door
        ↓
OpenObject fails
        ↓
spatial recovery / alternate pose
        ↓
Door opens
        ↓
cross / recompute reachable topology
        ↓
semantic target remains Apple
```

This removes a hidden one-shot `OpenObject` path that existed in v0.10.2.

## First required experiment: Door Level 0

Do not start Level 1 immediately after installation. Run one clean v0.10.3 calibration batch:

```bash
EPISODES=100 WORKERS=1 \
./scripts/run_training_wsl.sh \
  --curriculum door \
  --level 0
```

Then:

```bash
python scripts/inspect_skills.py
python scripts/inspect_actionability.py --worst 30
python -m json.tool state/metrics/training_summary.json
```

Desired evidence is not merely a high headline success rate. We specifically want at least one case where:

```text
bad initial pose
→ OpenObject failure
→ spatially different recovery pose
→ OpenObject success
→ recovery_successes increments
```

Recommended graduation gate:

```text
latest Level-0 door_open >= 97%
recovery_successes > 0 if bad poses were encountered
recovery_exhausted low / non-systematic
NorthArchive remains ~100%
EastExperiment no longer loses retries to one X/Z region
```

If the random run happens not to encounter a bad pose, repeat another 100 Level-0 episodes before concluding recovery is proven.

## Curriculum after Level 0

Only after the recovery evidence is healthy:

```bash
EPISODES=150 WORKERS=1 ./scripts/run_training_wsl.sh --curriculum door --level 1
EPISODES=200 WORKERS=1 ./scripts/run_training_wsl.sh --curriculum door --level 2
EPISODES=250 WORKERS=1 ./scripts/run_training_wsl.sh --curriculum door --level 3
EPISODES=300 WORKERS=1 ./scripts/run_training_wsl.sh --curriculum door --level 4
EPISODES=300 WORKERS=1 ./scripts/run_training_wsl.sh --curriculum door --level 5
```

Inspect between every level. Per-level skill statistics remain the graduation authority; cumulative easy-level history must not hide a harder-level regression.

## Manipulation generalization

Exact + X/Z actionability applies to:

```text
OpenObject
CloseObject
PickupObject
PutObject
ToggleObjectOn
ToggleObjectOff
```

Hard X/Z exclusion/radius recovery is strongest for moving open/close geometry. Pickup/toggle/place retain exact orientation evidence because camera/yaw can matter even at the same standing location.

After door Levels 0–4 are healthy, manipulation training can continue:

```bash
EPISODES=400 WORKERS=1 \
./scripts/run_training_wsl.sh \
  --curriculum pickup,place,toggle \
  --level 0
```

Then inspect object-specific competence and actionability:

```bash
python scripts/inspect_skills.py --object-details
python scripts/inspect_actionability.py --worst 40
```

## Replay and SFT

Scored stages are written to:

```text
state/replay/embodied_replay.jsonl
```

v0.10.3 schema:

```text
author.embodied.v0103
```

Controller-private telemetry includes exact/position keys, simulator errors, attempt sequence, and recovery summary. Only final simulator-verified successful demonstrations are exported to:

```text
state/datasets/embodied_sft.jsonl
```

## Multi-worker safety

Persistent A/B actionability is copied into each worker as a private baseline. At merge, only each worker's delta from that seed is added back. v0.10.3 performs this for both exact-pose and X/Z position statistics, preventing historical experience from being multiplied by worker count.

## Coverage metrics

Use:

```text
run_unique_cells
persistent_unique_cells
run_coverage_pct
```

`coverage_pct` remains a compatibility alias for current-run coverage and cannot exceed 100%.

## Live Escape gate

After Door Levels 0–4 pass, start the normal two-agent session and test only A first:

```text
/a Escape.
```

Desired motor behavior:

```text
Qwen selects Door
        ↓
actionability-aware approach
        ↓
OpenObject
   ├─ success → continue
   └─ failure → spatially different pose → retry
        ↓
verified isOpen=True
        ↓
CrossThreshold
        ↓
verified crossed=True
        ↓
Escape verifier succeeds
```

Qwen never needs X/Z coordinates or hinge-clearance reasoning.

## Do not reset

Do not use `--reset-memory` unless intentionally starting a new experimental lineage. v0.10.3 is designed to continue the existing A/B history and upgrade it with spatial actionability rather than discard it.
