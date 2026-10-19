# AUTHOR v0.10.12 Operator Guide

## Purpose

v0.10.12 is a surgical maintenance release over v0.10.10. It imports `facing_yaw` into `training.py`, which restores the intended Level-3/4 path-aligned spawn heading. No Unity rebuild is required.

The v0.10.10 FIFO `select()` timeout patch, simulator restart behavior, maneuverable-start filter, and all v0.10.9 room/stool geometry remain unchanged.

## Upgrade from v0.10.10

```bash
cd ~/projects
BACKUP=~/projects/author_v0.10.10_backup_$(date +%Y%m%d-%H%M%S)
mv ~/projects/author "$BACKUP"
unzip /mnt/c/Users/davei/Downloads/author_phase9_v0.10.12_facing_yaw_import_fix.zip -d ~/projects
cp -a "$BACKUP/state" ~/projects/author/
cd ~/projects/author
source ~/.venvs/author/bin/activate
python -m pip install -e ".[dev]"
pytest -q
```

Expected: `189 passed`.

Verify version:

```bash
python - <<'PY'
import agent_core
print(agent_core.__version__)
PY
```

Expected: `0.10.12`.

Do not use `--reset-memory`. The failed 0/60 v0.10.10 run did not complete worker merge, so preserve the existing state.

## Level-3 validation

```bash
EPISODES=60 WORKERS=1 \
./scripts/run_training_wsl.sh \
  --curriculum door \
  --level 3
```

Startup should report `version: 0.10.12`. There should be no `NameError: facing_yaw`. If Unity stalls, allow the configured 30-second FIFO timeout to fire before interrupting.
