# Upgrade: Continuous Brain v0.4.0 → v0.4.1

v0.4.1 fixes the L0 curriculum exposed by the first real AI2-THOR run. Preserve that run for comparison, but the recommended canonical v0.4.1 L0 branch starts again from the preserved v0.3 neural checkpoint.

## 1. Stop v0.4 cleanly

Use `Ctrl+C` so the checkpoint is saved.

## 2. Preserve the first real v0.4 embodied run

From Ubuntu-24.04:

```bash
cd ~/projects/continuous_brain_project

if [ -d state/embodied_brain ]; then
  mv state/embodied_brain state/embodied_brain_v040_first_real_run
fi
```

Do not delete it. It is a useful baseline.

## 3. Preserve the v0.4 source tree

```bash
cd ~/projects
mv continuous_brain_project continuous_brain_v0.4.0_preserved
```

## 4. Extract v0.4.1

Assuming the ZIP is in Windows Downloads:

```bash
unzip /mnt/c/Users/davei/Downloads/continuous_brain_v0.4.1.zip -d ~/projects
```

## 5. Reuse the working Python/PyTorch environment

```bash
cp -a \
  ~/projects/continuous_brain_v0.4.0_preserved/.venv \
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
```

Expected:

```text
0.4.1
```

## 6. Test

```bash
pytest -q
```

Expected release validation:

```text
20 passed
```

## 7. Start a clean corrected L0 branch

Use the preserved v0.3 checkpoint if available:

```bash
export THOR_EXECUTABLE="$HOME/projects/ai2thor-humanoid/builds/thor-discovery-Linux64/thor-discovery-Linux64"

continuous-brain \
  --environment thor \
  --scene FloorPlan1Discovery_physics \
  --thor-executable "$THOR_EXECUTABLE" \
  --state-dir state/embodied_brain_v041 \
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

After the first v0.4.1 checkpoint exists, omit `--bootstrap-checkpoint` on subsequent runs.

## 8. What to watch

For L0, the terminal should only show these brain intents:

```text
idle
turn_left
turn_right
```

You should never see:

```text
brain=advance
thor=MoveAhead
```

The agent's top-down position should remain fixed for each L0 episode while yaw changes.

The terminal now reports, for example:

```text
brain=turn_left thor=RotateLeft dBearing=+30.0 stagnation=0/60 Lstats=96.0%/med4.0
```

The browser dashboard displays allowed actions and rolling curriculum statistics.

## 9. Graduation target

Do not advance to L1 solely because a step count was reached. The default readiness target is:

- at least 100 recent L0 episodes
- at least 95% success
- median successful completion ≤ 6 steps
- timeout rate ≤ 2%

When those conditions are met, `graduation_ready` becomes true.

Then stop cleanly and resume the **same v0.4.1 state branch** with:

```bash
continuous-brain \
  --environment thor \
  --scene FloorPlan1Discovery_physics \
  --thor-executable "$THOR_EXECUTABLE" \
  --state-dir state/embodied_brain_v041 \
  --curriculum-level 1 \
  --goal-signal teacher-nav \
  --goal-dropout 0.0 \
  --vision-backbone pixel-stats \
  --visual \
  --open-browser
```
