# AUTHOR v0.10.6 Operator Guide

## Why this release exists

The real v0.10.5 Level-3 door curriculum advanced normally through episode 45 and then stopped producing replay rows. Interrupting the trainer showed the exact block:

```text
training._prepare_agent_start
  -> training._teleport
  -> Controller.step(action="TeleportFull")
  -> fifo_server.receive
  -> fifo_server._recv_message
  -> server_pipe.read(...)
```

This is simulator infrastructure, not embodied competence. No A/B reset is appropriate.

## What v0.10.6 does

Every shared-world simulator action is routed through an AUTHOR-side hard deadline on Linux/WSL. The default is 30 seconds. If Unity does not reply before that deadline:

1. `SimulatorStepTimeout` escapes local setup/action helpers.
2. The attempt is skipped and is **not** recorded as a skill failure.
3. The timed-out controller is considered protocol-desynchronized.
4. AUTHOR closes it and starts a fresh shared two-agent controller.
5. Existing in-memory worker skill/actionability/replay state is retained.
6. Training continues with the next raw setup attempt.

The worker writes `heartbeat.json` before and after simulator calls. Progress prints every 10 scored episodes.

## Upgrade from v0.10.5

Do not reset memory.

```bash
cd ~/projects
mv author author_v0.10.5_backup

unzip \
  /mnt/c/Users/davei/Downloads/author_phase9_v0.10.6_simulator_watchdog.zip \
  -d ~/projects

cp -a \
  ~/projects/author_v0.10.5_backup/state \
  ~/projects/author/

cd ~/projects/author
source ~/.venvs/author/bin/activate
python -m pip install -e ".[dev]"
pytest -q
```

Expected release test count is recorded in `RELEASE_VALIDATION.txt`.

Verify:

```bash
python - <<'PY'
import agent_core
print(agent_core.__version__)
PY
```

Expected:

```text
0.10.6
```

## Optional preflight

```bash
python -m agent_core.training \
  --config config.yaml \
  --scene FloorPlan1Discovery_physics \
  --episodes 1 \
  --workers 1 \
  --curriculum door \
  --level 3 \
  --dry-run
```

The plan should report:

```text
version: 0.10.6
trainer: watchdog_recoverable_curriculum
level: 3
simulator_step_timeout_seconds: 30.0
```

## Rerun Level 3

The interrupted v0.10.5 worker was never merged into global A/B checkpoints. Start a clean Level-3 batch from the graduated Level-2 state already present in `state/`:

```bash
EPISODES=250 WORKERS=1 \
./scripts/run_training_wsl.sh \
  --curriculum door \
  --level 3
```

New progress output should resemble:

```text
[worker 0][level 3] 10/250 scored | success 100.00% | skips 0 | timeouts 0 | restarts 0
```

If the old stall recurs, it should now recover automatically:

```text
[worker 0][level 3] simulator timeout during TeleportFull; restarting Unity
```

and later progress should continue.

## Heartbeat inspection

While training is active, from a second terminal:

```bash
python scripts/inspect_training_heartbeat.py
```

or continuously:

```bash
watch -n 2 python scripts/inspect_training_heartbeat.py
```

A heartbeat may look like:

```json
{
  "worker_id": 0,
  "level": 3,
  "episode": 46,
  "raw_attempt": 46,
  "skill": "door",
  "agent": "B",
  "status": "waiting_for_unity",
  "operation": "TeleportFull",
  "agent_id": 1,
  "restart_count": 0
}
```

## Configuration

`config.yaml`:

```yaml
training:
  simulator_step_timeout_seconds: 30.0
  simulator_start_timeout_seconds: 60.0
  progress_every_episodes: 10
```

Do not raise the step timeout merely to hide repeated hangs. A timeout should be rare; if it repeats at the same action/setup geometry, inspect the heartbeat and Unity log.

## State policy

- Never use `--reset-memory` for this upgrade.
- A and B remain independent.
- Existing Level-0/1/2 skills remain intact.
- Existing per-agent actionability remains intact.
- Timed-out simulator attempts are infrastructure skips, not competence failures.
- Raw completed worker rows before a restart remain part of that worker and merge normally when the worker finishes.
