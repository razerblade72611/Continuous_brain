# Upgrade v0.5.2 → v0.5.3

Preserve the v0.5.2 recovery-storm state as diagnostic evidence. Do not use it as the canonical L2 continuation.

```bash
Ctrl+C
cd ~/projects/continuous_brain_project
if [ -d state/embodied_brain_v052_spatial ]; then
  mv state/embodied_brain_v052_spatial state/embodied_brain_v052_recovery_storm_diagnostic
fi
```

Preserve the code and unpack v0.5.3:

```bash
cd ~/projects
mv continuous_brain_project continuous_brain_v0.5.2_preserved
unzip /mnt/c/Users/davei/Downloads/continuous_brain_v0.5.3.zip -d ~/projects
cp -a ~/projects/continuous_brain_v0.5.2_preserved/.venv ~/projects/continuous_brain_project/
cd ~/projects/continuous_brain_project
source .venv/bin/activate
python -m pip install -e . --no-deps
python -m pytest -q
```

Start a clean spatial branch from the protected L1-mastered v0.4.3 state:

```bash
mkdir -p state
cp -a \
  ~/projects/continuous_brain_v0.4.3_preserved/state/embodied_brain_v043_clean \
  state/embodied_brain_v053_spatial

export THOR_EXECUTABLE="$HOME/projects/ai2thor-humanoid/builds/thor-discovery-Linux64/thor-discovery-Linux64"
STATE_DIR=state/embodied_brain_v053_spatial ./scripts/run_l2_spatial_v053.sh
```

Expected behavior in native simulator mode:

- `dstale=1` may appear when depth becomes untrusted, but it should not by itself increment `recover=`.
- RGB freeze across actual camera motion still triggers `[sensor-recovery]`.
- Controller exceptions still trigger controller recovery.
- Repeated successful scene resets are bounded by cooldown and the soft-reset window.
- Map occupancy must not be erased by invalid depth.

Once L2 runs usefully for a meaningful window, development should advance to v0.6 semantic/hierarchical skills rather than endlessly tuning simulator infrastructure.
