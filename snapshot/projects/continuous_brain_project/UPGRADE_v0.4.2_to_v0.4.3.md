# Upgrade Continuous Brain v0.4.2 → v0.4.3

## 1. Stop and preserve the current diagnostic run

```bash
Ctrl+C
cd ~/projects/continuous_brain_project

if [ -d state/embodied_brain_v042 ]; then
  mv state/embodied_brain_v042 state/embodied_brain_v042_navigation_diagnostic
fi
```

## 2. Preserve the v0.4.2 source tree

```bash
cd ~/projects
mv continuous_brain_project continuous_brain_v0.4.2_preserved
```

## 3. Extract v0.4.3

```bash
unzip /mnt/c/Users/davei/Downloads/continuous_brain_v0.4.3.zip -d ~/projects
```

## 4. Reuse the working venv

```bash
cp -a \
  ~/projects/continuous_brain_v0.4.2_preserved/.venv \
  ~/projects/continuous_brain_project/

cd ~/projects/continuous_brain_project
source .venv/bin/activate
python -m pip install -e . --no-deps
```

## 5. Verify

```bash
python - <<'PY'
import continuous_brain
print(continuous_brain.__version__)
PY

python -m pytest -q
```

Expected:

```text
0.4.3
39 passed
```

## 6. Start a clean canonical embodied branch

Do not continue the known-bad L1 weights as the canonical experiment. Bootstrap neural weights from the preserved v0.3 brain and retrain corrected L0:

```bash
export THOR_EXECUTABLE="$HOME/projects/ai2thor-humanoid/builds/thor-discovery-Linux64/thor-discovery-Linux64"

continuous-brain \
  --environment thor \
  --scene FloorPlan1Discovery_physics \
  --thor-executable "$THOR_EXECUTABLE" \
  --state-dir state/embodied_brain_v043_clean \
  --bootstrap-checkpoint \
    ~/projects/continuous_brain_v0.3_preserved/state/my_brain/brain.pt \
  --curriculum-level 0 \
  --goal-signal teacher-nav \
  --goal-dropout 0.0 \
  --stagnation-steps 60 \
  --vision-backbone pixel-stats \
  --visual \
  --open-browser
```

Wait for the automatic mastered-skill message and then stop cleanly:

```text
[skill-mastered] L0 snapshot saved: .../skills/L0_mastered.pt
```

## 7. Continue the same brain into corrected L1

```bash
continuous-brain \
  --environment thor \
  --scene FloorPlan1Discovery_physics \
  --thor-executable "$THOR_EXECUTABLE" \
  --state-dir state/embodied_brain_v043_clean \
  --curriculum-level 1 \
  --goal-signal teacher-nav \
  --goal-dropout 0.0 \
  --stagnation-steps 60 \
  --rehearsal-fraction 0.20 \
  --nav-min-path-steps 2 \
  --nav-initial-path-steps 2 \
  --nav-max-path-steps 8 \
  --nav-success-radius 0.12 \
  --nav-alignment-gate-degrees 15 \
  --vision-backbone pixel-stats \
  --visual \
  --open-browser
```

No `--bootstrap-checkpoint` is used for this continuation.

## What to watch

A real L1 success should now be accompanied by physical translation. Watch these fields:

```text
L1goals=
trans=
blocked=
replans=
edges=
move=
dDist=
depth=
ok=
```

`L1goals` must not rise while `trans` remains zero.
