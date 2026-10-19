# AUTHOR v0.10.26 — Peer-Authoritative Neutral Parking

Level-4 toggle setup now parks the passive peer using peer-authoritative reachability, accepts an already-safe peer pose, and falls back to two-agent neutral normalization with explicit starvation diagnostics. The v0.10.24 broken-target guard and all v0.10.23 place/carried-route contracts are preserved.

# AUTHOR v0.10.25 — Peer-Neutral Toggle Route Qualification

AUTHOR v0.10.25 preserves the qualified Level-4 pickup/place stack and the v0.10.24 broken-toggle guard, while fixing stateful setup-search starvation in Level-4 toggle training.

Before one-door route qualification, the trainer now parks the passive agent in a neutral maneuverable cell. This prevents the previous scored episode's terminal peer pose from contaminating repeated GetReachablePositions/GetInteractablePoses queries when a setup is rejected and the scored episode number does not advance.

See `AUTHOR_v0.10.25_OPERATOR_README.md` for upgrade and qualification commands.
