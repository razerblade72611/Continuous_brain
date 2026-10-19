# Continuous Brain v0.4.0 — Embodied 3-D Branch

v0.4 is the first release that places the Continuous Brain behind an explicit embodiment boundary.
It preserves v0.3 as a useful baseline and adds a separate AI2-THOR path intended for simulation-to-real research.

## Added

- `AI2ThorEmbodiedEnv` with optional custom/local AI2-THOR executable support.
- RGB, depth and agent-pose ingestion from AI2-THOR events.
- A checkpoint-compatible **6 observation / 5 intent** embodiment interface.
- High-level intents: `idle`, `advance`, `turn_left`, `turn_right`, `interact`.
- 10-level embodied curriculum scaffold (orientation through multi-step interaction).
- Teacher navigation signal that can be progressively hidden with `--goal-dropout`.
- `--goal-signal sensor-only` mode to remove point-goal coordinates from policy observations.
- Depth-based local safety stop before forward motion.
- Action noise option for early sim-to-real robustness experiments.
- First-person RGB feed in the live browser monitor.
- Top-down diagnostic trace, yaw, horizon, depth sectors, task/curriculum metadata.
- `--bootstrap-checkpoint` to import neural weights into a fresh embodied state branch without overwriting the source state.
- `--additional-steps` for curriculum scripting.
- `run_thor.sh` and `scripts/run_thor_curriculum.sh`.

## Compatibility strategy

The recurrent core still receives six structured values and emits five learned high-level intents. This allows compatible v0.3 model tensors to be imported instead of immediately replacing every learned layer merely because the physical environment is more complex.

The six structured slots in the AI2-THOR adapter are:

1. local goal forward component (optionally masked)
2. local goal right component (optionally masked)
3. center depth clearance
4. left-right depth clearance difference
5. last action success
6. last extrinsic reward

The visual foundation embedding remains a separate residual pathway through the trainable adapter introduced in v0.3.

## Important limitation

v0.4 is an embodied research scaffold, not a real-world autonomous robot controller. The policy has not yet learned SLAM, object permanence, a metric map, manipulation geometry, or calibrated real sensors. The safety stop is a simulator-side guard, not a sufficient physical safety system.
