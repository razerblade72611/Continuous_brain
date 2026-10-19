# Upgrade v0.4.1 → v0.4.2

v0.4.2 preserves the neural architecture but changes curriculum statistics, navigation supervision and replay behavior.

## 1. Stop v0.4.1 cleanly

Press `Ctrl+C` and allow the checkpoint to save.

## 2. Preserve the v0.4.1 L1 diagnostic branch

From the old project directory:

```bash
cd ~/projects/continuous_brain_project

if [ -d state/embodied_brain_v041 ]; then
  mv state/embodied_brain_v041 state/embodied_brain_v041_l1_diagnostic
fi
```

If you already renamed it earlier, leave it as-is.

## 3. Preserve the old code and extract v0.4.2

```bash
cd ~/projects
mv continuous_brain_project continuous_brain_v0.4.1_preserved
unzip /mnt/c/Users/davei/Downloads/continuous_brain_v0.4.2.zip -d ~/projects
```

## 4. Reuse the existing venv

```bash
cp -a ~/projects/continuous_brain_v0.4.1_preserved/.venv \
      ~/projects/continuous_brain_project/

cd ~/projects/continuous_brain_project
source .venv/bin/activate
python -m pip install -e . --no-deps
python -m pytest -q
```

Expected: `28 passed`.

## 5. Create a new v0.4.2 state branch

If the diagnostic state is inside the preserved v0.4.1 project:

```bash
cp -a \
  ~/projects/continuous_brain_v0.4.1_preserved/state/embodied_brain_v041_l1_diagnostic \
  ~/projects/continuous_brain_project/state/embodied_brain_v042
```

If your diagnostic state has another name, copy that directory instead.

This retains the learned neural weights and episodic memory while v0.4.2 migrates the old mixed curriculum statistics into legacy diagnostic storage and starts clean per-level L0/L1 windows.

## 6. Start corrected L1

```bash
export THOR_EXECUTABLE="$HOME/projects/ai2thor-humanoid/builds/thor-discovery-Linux64/thor-discovery-Linux64"

continuous-brain \
  --environment thor \
  --scene FloorPlan1Discovery_physics \
  --thor-executable "$THOR_EXECUTABLE" \
  --state-dir state/embodied_brain_v042 \
  --curriculum-level 1 \
  --goal-signal teacher-nav \
  --goal-dropout 0.0 \
  --stagnation-steps 60 \
  --rehearsal-fraction 0.20 \
  --nav-initial-path-steps 2 \
  --nav-max-path-steps 8 \
  --vision-backbone pixel-stats \
  --visual \
  --open-browser
```

Do not use `--bootstrap-checkpoint` when continuing from the copied v0.4.1 state directory.

## What should look different

- L1 statistics start from 0 trustworthy L1 episodes rather than inheriting L0 success.
- Epsilon should jump back toward the level-specific floor (default around 0.35 at the start of fresh L1 evidence).
- Terminal lines should show `wp=i/n`.
- Correct turns can produce positive `dBearing` reward even while `dDist=0`.
- Successful forward movement should show nonzero `move=` and often positive `dDist=`.
- Some episodes will explicitly report `rehearse=L0`.
- `SafetyStop`/blocked advances receive a stronger penalty.
