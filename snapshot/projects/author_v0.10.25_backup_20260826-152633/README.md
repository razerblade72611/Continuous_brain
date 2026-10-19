# AUTHOR v0.10.25 — Peer-Neutral Toggle Route Qualification

AUTHOR v0.10.25 preserves the qualified Level-4 pickup/place stack and the v0.10.24 broken-toggle guard, while fixing stateful setup-search starvation in Level-4 toggle training.

Before one-door route qualification, the trainer now parks the passive agent in a neutral maneuverable cell. This prevents the previous scored episode's terminal peer pose from contaminating repeated GetReachablePositions/GetInteractablePoses queries when a setup is rejected and the scored episode number does not advance.

See `AUTHOR_v0.10.25_OPERATOR_README.md` for upgrade and qualification commands.
