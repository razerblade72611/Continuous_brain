# AUTHOR v0.10.12 Operator Guide

v0.10.12 is a transport-recovery maintenance release over v0.10.11. It preserves the verified stool-free, maneuver-clear Discovery geometry and all learned A/B state.

## What changed

- `BrokenPipeError`, connection reset/abort, EOF, and FIFO-close/short-read conditions are classified as simulator transport failures.
- A transport failure is unscored, immediately discards the poisoned controller, restarts Unity, reinitializes both adapters, and continues training.
- `SharedThorWorld.close()` detaches the controller before stopping it and ignores an already-broken transport during cleanup.
- Training summaries now expose `simulator_timeouts`, `simulator_transport_failures`, and `simulator_restarts`.
- Level-3/4 maneuverable-start and path-aligned yaw behavior from v0.10.11 is unchanged.
- No Unity rebuild is required.

## Upgrade without resetting learning

```bash
cd ~/projects
BACKUP=~/projects/author_v0.10.11_backup_$(date +%Y%m%d-%H%M%S)
mv ~/projects/author "$BACKUP"
unzip /mnt/c/Users/davei/Downloads/author_phase9_v0.10.12_transport_recovery.zip -d ~/projects
cp -a "$BACKUP/state" ~/projects/author/
cd ~/projects/author
source ~/.venvs/author/bin/activate
python -m pip install -e ".[dev]"
pytest -q
```

Do not use `--reset-memory`.

## Validate Level 3

```bash
EPISODES=60 WORKERS=1 \
./scripts/run_training_wsl.sh \
  --curriculum door \
  --level 3
```

Startup should report version `0.10.12` and trainer `transport_recovering_fifo_maneuverable_curriculum`.

If Unity dies mid-run, expect a line like:

```text
simulator transport failure during TeleportFull; restarting Unity
```

That setup is unscored. Training should then resume and the final summary should include transport/restart counts.
