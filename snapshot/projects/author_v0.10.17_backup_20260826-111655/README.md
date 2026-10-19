# AUTHOR v0.10.17 — Canonical Scene Reset

v0.10.17 prevents Level-4 pickup curriculum state drift by restoring the canonical full pose of every movable/pickupable object between independent lessons. It removes the old post-pickup `PlaceObjectAtPoint` cleanup and retains the v0.10.16 semantic-route, peer-safety, dynamic-replan, FIFO-watchdog, and transport-recovery behavior.

See `AUTHOR_v0.10.17_OPERATOR_README.md`.
