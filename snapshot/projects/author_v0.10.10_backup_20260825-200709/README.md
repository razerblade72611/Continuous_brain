# AUTHOR v0.10.10 — FIFO-Bounded Maneuverable Starts

v0.10.10 preserves the v0.10.9 runtime-sanitized maneuver-clear Discovery geometry and hardens two Level-3 infrastructure paths:

1. AI2-THOR 4.3.x FIFO reads are guarded at the file-descriptor layer with `select()` deadlines, so a wedged Unity response raises `SimulatorStepTimeout` without depending on `SIGALRM` interrupting `read()` on WSL.
2. Level-3/4 trainer starts are filtered for local maneuverability (at least two cardinal neighbors) and are oriented along the first shortest-path segment instead of using a random yaw.

Persistent A/B state, replay, actionability memory, stool removal, room keep-outs, exact-pose interaction verification, and earlier curriculum fixes are preserved.

See `AUTHOR_v0.10.10_OPERATOR_README.md` for upgrade and validation commands.
