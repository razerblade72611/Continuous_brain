# AUTHOR v0.10.5 — CrossThreshold Restoration + State Repair

v0.10.5 is a surgical reliability release on top of v0.10.4.

The v0.10.4 pose-authoritative Level-1 fix worked: the new run achieved 75/75 approaches for A and 74/75 for B, with 149/149 successful door openings. However, the v0.10.4 refactor accidentally omitted the live `_cross_threshold_for_adapter` helper (and the related `FaceTarget` helper) from `thor_adapter.py`. Every new `CrossThreshold` dispatch therefore failed with a `NameError`, creating synthetic `door_cross` failures and making the full Level-1 run appear 0% successful.

v0.10.5:

- restores physical `CrossThreshold` execution,
- restores live `FaceTarget` dispatch,
- preserves the v0.10.4 pose-authoritative NavigateTo behavior,
- adds live adapter-dispatch regression tests,
- adds an idempotent v0.10.4 false-cross-failure repair utility,
- preserves raw replay rows as audit evidence,
- keeps A/B persistent state and actionability memory intact.

See `AUTHOR_v0.10.5_OPERATOR_README.md` for upgrade, repair, and validation commands.
