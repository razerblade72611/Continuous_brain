# AUTHOR v0.10.4 — Pose-Authoritative Short-Range Navigation

v0.10.4 fixes the Level-1 verifier mismatch exposed by the 150-episode door-approach diagnostic.

## v0.10.4 focus

- `NavigateTo` success is now verified from explicit controller pose-goal evidence, not object-transform-center distance.
- Object-center distance remains diagnostic only for pose-aware navigation.
- Staged curriculum Levels 1–3 pin the exact `GetInteractablePoses` pose used to construct each episode.
- Controller telemetry distinguishes `interaction_pose_reached`, exhausted replans, unavailable pinned poses, blocked-door failures, and step-budget exhaustion.
- v0.10.3 spatial actionability/recovery remains intact and state-compatible.
- Coordinates remain controller-private and are not exposed to Qwen.

See `AUTHOR_v0.10.4_OPERATOR_README.md` for upgrade and validation commands.
