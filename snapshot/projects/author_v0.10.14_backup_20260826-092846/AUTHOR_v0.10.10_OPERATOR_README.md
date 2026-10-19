# AUTHOR v0.10.10 Operator Guide

## Purpose

This release fixes the remaining Level-3 `TeleportFull` stall seen under WSL with AI2-THOR 4.3.x and avoids trainer starts in dead-end/corner cells.

### FIFO watchdog

AI2-THOR 4.3.x performs blocking FIFO reads. v0.10.10 patches the runtime FIFO receiver before Controller startup and waits on the FIFO file descriptor with `select()` using AUTHOR's configured simulator timeout. A stalled header or body read raises `SimulatorStepTimeout`, allowing the worker to mark that setup unscored, restart Unity, and continue.

The existing SIGALRM wrapper remains only as a fallback; correctness no longer depends on it interrupting a blocking FIFO read.

### Maneuverable starts

For Levels 3 and 4, start candidates must have at least two cardinal reachable neighbors when such candidates are available. Candidates with greater local degree are preferred. The setup yaw faces the first segment of the verified shortest path rather than being randomly selected.

This is trainer setup hygiene, not semantic leakage: the semantic planner still receives no coordinates or geometry.

## Upgrade

```bash
cd ~/projects
BACKUP=~/projects/author_v0.10.9_backup_$(date +%Y%m%d-%H%M%S)
mv ~/projects/author "$BACKUP"
unzip /mnt/c/Users/davei/Downloads/author_phase9_v0.10.10_fifo_maneuver_watchdog.zip -d ~/projects
cp -a "$BACKUP/state" ~/projects/author/
cd ~/projects/author
source ~/.venvs/author/bin/activate
python -m pip install -e ".[dev]"
pytest -q
```

Expected: `187 passed`.

Do not use `--reset-memory`.

## Verify version

```bash
python - <<'PY'
import agent_core
print(agent_core.__version__)
PY
```

Expected: `0.10.10`.

## Geometry smoke test

No Unity rebuild is required when upgrading from the already-verified v0.10.9 Discovery player because v0.10.10 changes Python trainer/runtime communication, not Unity geometry.

```bash
python scripts/smoke_test_discovery_world.py \
  --thor-executable \
  /home/davei/projects/ai2thor-humanoid/builds/thor-discovery-Linux64/thor-discovery-Linux64
```

The verified v0.10.9 world should still report `Stool objects: []` and `PASS`.

## Level-3 validation

Start with 60 episodes:

```bash
EPISODES=60 WORKERS=1 \
./scripts/run_training_wsl.sh \
  --curriculum door \
  --level 3
```

Expected startup metadata:

- version `0.10.10`
- trainer `fifo_bounded_maneuverable_start_curriculum`
- simulator step timeout `30.0`

If Unity fails to reply, wait for AUTHOR to print a simulator timeout/restart message. Do not press Ctrl+C before the 30-second configured deadline when testing watchdog recovery.

After a clean 60-run:

```bash
python scripts/inspect_skills.py
python scripts/inspect_actionability.py --worst 30
python -m json.tool state/metrics/training_summary.json
```
