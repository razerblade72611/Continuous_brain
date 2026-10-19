# AUTHOR v0.10.11 — Facing-Yaw Import Fix

This maintenance release preserves v0.10.10 FIFO-bounded simulator communication, maneuverable Level-3/4 start filtering, and the verified v0.10.9 Discovery geometry.

It fixes a v0.10.10 packaging regression where `training.py` called `facing_yaw()` without importing it from `agent_core.navigation`, causing every Level-3 setup attempt to skip with `NameError: name 'facing_yaw' is not defined`.

See `AUTHOR_v0.10.11_OPERATOR_README.md` for upgrade and validation commands.
