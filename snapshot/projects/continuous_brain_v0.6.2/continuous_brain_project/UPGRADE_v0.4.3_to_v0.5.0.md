# Upgrade: Continuous Brain v0.4.3 → v0.5.0

Your clean v0.4.3 branch already has mastered L0 and L1 snapshots. Preserve that branch unchanged and create a new v0.5 spatial branch from it.

## 1. Stop the current v0.4.3 process

```bash
Ctrl+C
```

## 2. Preserve the v0.4.3 project

```bash
cd ~/projects
mv continuous_brain_project continuous_brain_v0.4.3_preserved
```

## 3. Extract v0.5.0

```bash
unzip \
  /mnt/c/Users/davei/Downloads/continuous_brain_v0.5.0.zip \
  -d ~/projects
```

## 4. Reuse the known-working virtual environment

```bash
cp -a \
  ~/projects/continuous_brain_v0.4.3_preserved/.venv \
  ~/projects/continuous_brain_project/

cd ~/projects/continuous_brain_project
source .venv/bin/activate
python -m pip install -e . --no-deps
```

Verify:

```bash
python - <<'PY'
import continuous_brain
print(continuous_brain.__version__)
PY
python -m pytest -q
```

Expected:

```text
0.5.0
50 passed
```

## 5. Branch from the L1-mastered state

Do not overwrite the v0.4.3 canonical state.

```bash
mkdir -p ~/projects/continuous_brain_project/state

cp -a \
  ~/projects/continuous_brain_v0.4.3_preserved/state/embodied_brain_v043_clean \
  ~/projects/continuous_brain_project/state/embodied_brain_v050_spatial
```

The copied state should already contain:

```text
skills/L0_mastered.pt
skills/L1_mastered.pt
brain.pt
brain.previous.pt
replay.pt
```

Verify before training:

```bash
python scripts/verify_state.py state/embodied_brain_v050_spatial
```

## 6. Run L2 spatial cognition

```bash
export THOR_EXECUTABLE="$HOME/projects/ai2thor-humanoid/builds/thor-discovery-Linux64/thor-discovery-Linux64"

STATE_DIR=state/embodied_brain_v050_spatial \
  ./scripts/run_l2_spatial_v05.sh
```

The script uses:

```text
curriculum          L2
waypoint source     internal spatial map
rehearsal           20% earlier skills
spatial map cell    0.25 m
minimum real moves  4
arrival tolerance   0.18 m
stop on mastery     enabled
```

Expected L2 telemetry includes:

```text
level=2 spatial-map
task=spatial_point
L2goals=...
trans=...
map=known/total
occ=...
plans=...
pfail=...
```

The important distinction is that `waypoint_source` is `spatial-map`, not `teacher-reachable-grid`.

## 7. Let the script stop itself

When L2 satisfies its graduation window, v0.5 prints:

```text
[skill-mastered] L2 snapshot saved: .../skills/L2_mastered.pt
[stop-on-mastery] L2 graduation confirmed; saving and exiting cleanly.
```

Then verify:

```bash
python scripts/verify_state.py state/embodied_brain_v050_spatial
```

## Rollback

If L2 behaves badly, stop it and preserve the experimental branch. The untouched v0.4.3 project/state and its `L1_mastered.pt` remain your known-good rollback point.
