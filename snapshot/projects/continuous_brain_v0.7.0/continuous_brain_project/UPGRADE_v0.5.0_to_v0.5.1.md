# Upgrade: Continuous Brain v0.5.0 → v0.5.1

## Important

Preserve the current L2 run as a diagnostic. Because that run accumulated invalid depth and degraded occupancy evidence, do not use it as the canonical v0.5.1 spatial-training branch. Start the corrected L2 branch from the preserved L1-mastered v0.4.3 state.

## Install

```bash
Ctrl+C
cd ~/projects/continuous_brain_project
if [ -d state/embodied_brain_v050_spatial ]; then
  mv state/embodied_brain_v050_spatial state/embodied_brain_v050_sensor_diagnostic
fi

cd ~/projects
mv continuous_brain_project continuous_brain_v0.5.0_preserved
unzip /mnt/c/Users/davei/Downloads/continuous_brain_v0.5.1.zip -d ~/projects
cp -a ~/projects/continuous_brain_v0.5.0_preserved/.venv ~/projects/continuous_brain_project/
cd ~/projects/continuous_brain_project
source .venv/bin/activate
python -m pip install -e . --no-deps
python -m pytest -q
```

Expected: `56 passed`.

## Create a clean corrected L2 state

```bash
mkdir -p state
cp -a \
  ~/projects/continuous_brain_v0.4.3_preserved/state/embodied_brain_v043_clean \
  state/embodied_brain_v051_spatial
```

If you made the full L1-mastered backup, that is equally suitable as the source.

## Run

```bash
export THOR_EXECUTABLE="$HOME/projects/ai2thor-humanoid/builds/thor-discovery-Linux64/thor-discovery-Linux64"
STATE_DIR=state/embodied_brain_v051_spatial ./scripts/run_l2_spatial_v051.sh
```

Watch the new terminal diagnostics: `rgb=<hash>/<diff> stale=0 dvalid=<fraction>`, plus `hard=<count>`. A true frozen RGB stream across movement/rotation is automatically detected and the episode is reset.
