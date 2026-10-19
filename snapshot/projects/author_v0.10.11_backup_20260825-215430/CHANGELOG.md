## 0.10.11

- Fixes Level-3/4 setup `NameError` by importing `facing_yaw` from `agent_core.navigation`.
- Adds an end-to-end regression for maneuverable Level-3 start selection through path-aligned yaw and teleport.
- Preserves FIFO select watchdog, recoverable simulator restarts, maneuverable-start filtering, and Discovery geometry unchanged.

# Changelog

## 0.10.10
- Replaced dependence on signal interruption of AI2-THOR 4.3.x FIFO reads with file-descriptor `select()` deadlines.
- FIFO header and body reads now raise `SimulatorStepTimeout` when Unity does not reply within AUTHOR's configured timeout.
- Configures per-action operation names in the FIFO watchdog for actionable heartbeat/error telemetry.
- Level-3/4 start sampling rejects dead-end cells when maneuverable alternatives exist.
- Level-3/4 starts face the first verified path segment instead of a random yaw.
- Preserves v0.10.9 maneuver-clear room geometry and runtime stool sanitizer.
- Adds v0.10.10 regression coverage for FIFO deadlines and maneuverability filtering.
