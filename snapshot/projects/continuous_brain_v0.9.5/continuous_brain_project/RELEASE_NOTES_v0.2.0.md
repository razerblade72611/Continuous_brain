# Continuous Brain v0.2.0 Release Notes

## New
- Dependency-free browser visualization (`--visual`).
- World canvas with actual agent, target, recent trail, and world-model predicted position.
- Live reward, goal, energy, distance, epsilon, surprise, replay, memory, and loss telemetry.
- Live Q-value display and sampled recurrent-state activity.
- Pause/resume and speed controls from the browser.
- Optional `--open-browser` WSL/desktop convenience.
- `run_visual.sh` helper.
- Exact v0.1 -> v0.2 upgrade guide.

## Compatibility
The neural network architecture, optimizer schema, checkpoint keys, environment observation dimensions, and persistent episodic database format remain compatible with v0.1.0.

The replay buffer remains transient and is rebuilt after each process restart; this was also true in v0.1.

## Validation
- Python compile check passed.
- 4 unit/integration tests passed.
- Browser visualizer `/`, `/state`, and `/health` endpoints tested.
- Visualizer pause/speed state controls tested.
- 120-step v0.2 visual smoke run completed and checkpointed.
- Genuine v0.1 checkpoint created at step 120 and resumed successfully under v0.2 through step 150.
