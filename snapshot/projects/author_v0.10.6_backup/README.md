# AUTHOR v0.10.6 — Simulator Watchdog + Recoverable Setup

v0.10.6 hardens AUTHOR against blocking AI2-THOR FIFO calls discovered during the real Level-3 door curriculum.

The Level-3 run progressed normally through episode 45, then blocked inside trainer `TeleportFull` while Python waited in AI2-THOR's FIFO `server_pipe.read(...)`. The installed behavior is consistent with the ai2thor 4.3.x FIFO path, where a missing Unity response can leave the Python process waiting indefinitely.

v0.10.6:

- adds an AUTHOR-side POSIX hard deadline around blocking simulator calls,
- treats simulator timeouts as infrastructure/setup failures rather than A/B competence failures,
- discards and restarts a timed-out shared two-agent Unity controller before continuing,
- preserves completed worker-local replay/skill state across simulator restarts,
- records worker heartbeat state with episode, raw attempt, skill, agent, action, and restart count,
- prints progress every 10 scored episodes by default,
- advances embodied replay/SFT/trainer metadata to v0.10.6,
- preserves v0.10.5 CrossThreshold/FaceTarget restoration, pose-authoritative navigation, actionability memory, and all persistent A/B state.

See `AUTHOR_v0.10.6_OPERATOR_README.md` for upgrade and Level-3 rerun instructions.
