# Upgrade: Continuous Brain v0.3 → v0.4

## Preserve v0.3

Do not overwrite the v0.3 directory or its state. Keep it as the 2-D baseline.

Example:

```bash
cd ~/projects
mv continuous_brain_project continuous_brain_v0.3_preserved
unzip /mnt/c/Users/$USER/Downloads/continuous_brain_v0.4.0.zip -d ~/projects
```

If your Windows username differs from your Linux username, use the actual Windows Downloads path.

## Reuse the existing Python environment

```bash
cp -a ~/projects/continuous_brain_v0.3_preserved/.venv \
      ~/projects/continuous_brain_project/
cd ~/projects/continuous_brain_project
source .venv/bin/activate
python -m pip install -e . --no-deps
pytest -q
```

This avoids replacing a deliberately selected PyTorch build.

## Keep the old brain and create a new embodied branch

Do **not** copy the old state directory on top of the new one. Instead bootstrap the neural weights once:

```bash
continuous-brain \
  --environment thor \
  --scene FloorPlan1Discovery_physics \
  --thor-executable "$THOR_EXECUTABLE" \
  --state-dir state/embodied_brain \
  --bootstrap-checkpoint ../continuous_brain_v0.3_preserved/state/my_brain/brain.pt \
  --curriculum-level 0 \
  --vision-backbone pixel-stats \
  --visual
```

`--bootstrap-checkpoint` imports compatible model tensors only. It deliberately resets counters, replay, optimizer state and episodic memory in the new embodied branch.

After the first save, omit `--bootstrap-checkpoint`; the embodied branch loads its own checkpoint normally.

## AI2-THOR Python package

The custom Unity executable and the Python `ai2thor` controller package are separate pieces. If the active venv does not already have the compatible controller package used by your fork, install/provide it before running `--environment thor`.

The project also declares an optional convenience extra:

```bash
python -m pip install -e '.[thor]' --no-deps
```

Use the controller version appropriate for your AI2-THOR fork rather than blindly replacing a known-working installation.
