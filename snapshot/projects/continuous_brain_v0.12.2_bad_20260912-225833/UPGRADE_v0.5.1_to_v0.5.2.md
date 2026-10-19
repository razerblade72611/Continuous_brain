# Upgrade v0.5.1 → v0.5.2

The v0.5.1 sensor-freeze diagnostic branch should be preserved, not continued as the canonical L2 state. Its L2 rolling statistics include infrastructure failures from before quarantine was implemented.

## Install

```bash
Ctrl+C
cd ~/projects/continuous_brain_project

if [ -d state/embodied_brain_v051_spatial ]; then
  mv state/embodied_brain_v051_spatial state/embodied_brain_v051_sensor_freeze_diagnostic
fi

cd ~/projects
mv continuous_brain_project continuous_brain_v0.5.1_preserved
unzip /mnt/c/Users/davei/Downloads/continuous_brain_v0.5.2.zip -d ~/projects
cp -a ~/projects/continuous_brain_v0.5.1_preserved/.venv ~/projects/continuous_brain_project/
cd ~/projects/continuous_brain_project
source .venv/bin/activate
python -m pip install -e . --no-deps
python -m pytest -q
```

Expected version: `0.5.2`.

## Recommended clean L2 branch

Recreate L2 from the known-good v0.4.3 L1-mastered state so old sensor-fault curriculum evidence does not follow the canonical branch:

```bash
mkdir -p ~/projects/continuous_brain_project/state
cp -a \
  ~/projects/continuous_brain_v0.4.3_preserved/state/embodied_brain_v043_clean \
  ~/projects/continuous_brain_project/state/embodied_brain_v052_spatial

export THOR_EXECUTABLE="$HOME/projects/ai2thor-humanoid/builds/thor-discovery-Linux64/thor-discovery-Linux64"
STATE_DIR=state/embodied_brain_v052_spatial ./scripts/run_l2_spatial_v052.sh
```

## Expected recovery message

If the custom THOR sensor freezes, training should print either:

```text
[sensor-recovery] recovered via scene-reset; quarantined fault tick and resuming.
```

or:

```text
[sensor-recovery] recovered via controller-restart; quarantined fault tick and resuming.
```

If verification cannot recover the stream, v0.5.2 saves and exits cleanly. Do not override that fail-closed behavior for canonical training.
