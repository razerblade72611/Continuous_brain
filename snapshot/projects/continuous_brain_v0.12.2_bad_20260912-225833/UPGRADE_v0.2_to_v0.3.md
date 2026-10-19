# Upgrade: Continuous Brain v0.2 -> v0.3

v0.3 is designed to migrate a v0.2 brain rather than discard it.

## Safe upgrade

Stop v0.2 with Ctrl+C so it writes its checkpoint.

```bash
cd ~/projects
mv continuous_brain_project continuous_brain_v0.2_backup
unzip /mnt/c/Users/davei/Downloads/continuous_brain_v0.3.0.zip -d ~/projects
```

Reuse the known-good virtual environment:

```bash
cp -a ~/projects/continuous_brain_v0.2_backup/.venv \
      ~/projects/continuous_brain_project/
```

Restore the learned state:

```bash
cp -a ~/projects/continuous_brain_v0.2_backup/state \
      ~/projects/continuous_brain_project/
```

Activate and refresh the editable install without replacing PyTorch:

```bash
cd ~/projects/continuous_brain_project
source .venv/bin/activate
python -m pip install -e . --no-deps
```

Validate:

```bash
python - <<'PY'
import continuous_brain
import torch
print("Continuous Brain:", continuous_brain.__version__)
print("PyTorch:", torch.__version__)
print("GPU available:", torch.cuda.is_available())
PY
pytest -q
```

## First migration run

Use the lightweight visual path first:

```bash
continuous-brain \
  --state-dir state/my_brain \
  --visual \
  --vision-backbone pixel-stats
```

You should see a message similar to:

```text
Checkpoint migration: preserved compatible v0.1/v0.2 weights; new=... ignored=0
Checkpoint migration: optimizer state reset for new v0.3 parameters.
```

This is expected. The old learned network weights are retained; only the new visual adapter/gate is uninitialized from prior training.

## Enable DINOv3 later

After accepting the Meta DINOv3 Hugging Face access terms:

```bash
python -m pip install "transformers>=4.56" pillow
continuous-brain --state-dir state/my_brain --visual \
  --vision-backbone dinov3-vits16 --vision-device cpu --vision-every 5
```

The DINOv3 backbone is frozen by default.
