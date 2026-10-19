# AUTHOR v0.10.19 — Portal-Center Crossing + Pose-Qualified Place

This release repairs the v0.10.18 Level-4 place regression by crossing through the physical closed-door panel center rather than the Door hinge, allowing later target-component motion to authoritatively prove a cross, and preflighting/pinning the exact `PutObject(forceAction=False)` interaction pose used by scored place lessons.

See `AUTHOR_v0.10.19_OPERATOR_README.md`.


A one-time idempotent `scripts/repair_v01018_place_regression.py` migration removes only known v0.10.18 regression labels from persistent skill counters while preserving raw replay.
