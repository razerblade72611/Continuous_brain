# AUTHOR v0.10.20 — Carried-Object Door Clearance

v0.10.20 is a narrow Level-4 place transport correction on top of v0.10.19's working placement contract.

## Root cause fixed

The completed v0.10.19 Level-4 place run produced 31/60 successes. All 29 failures occurred before release: 28 at door crossing and one at door approach. The simulator errors showed two carried-object failure modes:

1. 18 failures were 90-degree body rotations while holding an object near the doorway jamb.
2. 10 failures were off-center east-portal moves into `EastPortalBoundary_Upper/Lower`.
3. One additional cross primitive exception occurred before route continuation.

`PutObject` itself had zero failures in the v0.10.19 batch and is intentionally unchanged.

## Motor changes

- `path_to_actions` now supports `face_target=False`; CrossThreshold uses it for source-side staging so a zero-vector staging target cannot inject a meaningless final yaw at the jamb.
- When carrying an object, closed-door selection prefers an interaction pose that can be entered already facing the door, avoiding a last-second held-object turn at the door approach pose.
- CrossThreshold chooses source and destination cells on the same portal-center grid lane.
- Source staging happens 0.85–1.25 m from the portal depending on the held object's AABB footprint.
- All body alignment happens at that safe staging depth.
- Portal traversal itself is MoveAhead-only: no body rotation is allowed inside the doorway clearance zone.
- The v0.10.19 exact place-pose preflight, openable-receptacle handling, canonical scene reset, transport watchdogs, semantic target preservation, and no-normal-runtime-teleport rules are retained.

## State repair

The completed v0.10.19 run merged 29 known transport-regression failures into persistent A/B counters. After installing v0.10.20 run:

```bash
python scripts/repair_v01019_place_transport_regression.py --state-dir state
```

The repair is idempotent. It backs up A/B checkpoints, subtracts only replay rows carrying explicit v0.10.19 held-item / PortalBoundary / cross-primitive transport evidence, and leaves raw replay untouched.

## Upgrade

Preserve the existing `state/` directory. Do not use `--reset-memory`.

```bash
cd ~/projects
BACKUP=~/projects/author_v0.10.19_backup_$(date +%Y%m%d-%H%M%S)
mv ~/projects/author "$BACKUP"
unzip /mnt/c/Users/davei/Downloads/author_phase9_v0.10.20_carried_object_clearance.zip -d ~/projects
cp -a "$BACKUP/state" ~/projects/author/
cd ~/projects/author
source ~/.venvs/author/bin/activate
python -m pip install -e ".[dev]"
pytest -q
```

Expected tests: `231 passed` and version `0.10.20`.

Then repair the v0.10.19 bad labels and rerun the Level-4 place pilot:

```bash
python scripts/repair_v01019_place_transport_regression.py --state-dir state

EPISODES=60 WORKERS=1 \
./scripts/run_training_wsl.sh \
  --curriculum place \
  --level 4
```

Expected startup trainer: `carried_object_clearance_curriculum`.

No Unity rebuild is required.
