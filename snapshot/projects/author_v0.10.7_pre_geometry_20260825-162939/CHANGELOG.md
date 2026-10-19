# v0.10.7 — Selected Event Contract Fix

- Fixes a v0.10.6 regression where trainer `_teleport()` and `_admin_step()` called `SharedThorWorld.step_for()` and then selected `agent_id` a second time from the already-selected Event.
- The double-selection caused `IndexError: list index out of range` when trainer infrastructure targeted agent B, especially `_park_peer()` in Level 3.
- Keeps v0.10.6 simulator watchdog/restart behavior unchanged.
- Adds agent-B regression tests for trainer teleport and admin actions.
- Advances embodied replay/SFT release metadata to v0.10.7.
