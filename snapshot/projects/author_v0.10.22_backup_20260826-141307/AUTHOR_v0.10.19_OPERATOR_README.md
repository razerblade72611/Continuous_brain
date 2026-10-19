# AUTHOR v0.10.19 — Portal-Center Crossing + Pose-Qualified Place

v0.10.19 is a narrow Level-4 place correction on top of the stable v0.10.17/v0.10.18 canonical-scene and transport work.

It fixes two regressions isolated by the v0.10.18 place qualification:

1. Door `position` is a hinge/pivot in the Discovery build, not the architectural opening center. Crossing now captures the closed Door AABB center, stages on the source side near that portal centerline, and targets a cell deeper in the newly connected component.
2. GetReachablePositions is treated as a position set, not a guaranteed edge graph. The cross helper no longer selects arbitrary cardinal-adjacent source/outside cells along thin walls.
3. A provisional `door_cross=false` is promoted if later embodied motion reaches the original semantic target outside the closed-door source component. This prevents a successful far-side target approach from being audited as a failed cross.
4. Level-4 place now preflights `PutObject(forceAction=False)` from the exact candidate interaction pose and pins the scored approach to that verified controller-private pose.
5. Openable receptacle handling and pair-level physical feasibility checks from v0.10.18 remain in place.
6. Replay/SFT schemas are `author.embodied.v01019` / `author.embodied_sft.v01019`.

No Unity rebuild is required. Preserve `state/`; do not use `--reset-memory`.

## Repair the completed v0.10.18 regression labels

After preserving the v0.10.18 state directory, run this once before new training:

```bash
python scripts/repair_v01018_place_regression.py --state-dir state
```

The repair is idempotent. It backs up A/B skill checkpoints, removes only replay-matched v0.10.18 Level-4 place bad-cross / invalid-release failure labels from skill counters, and leaves raw replay untouched.

## Validate

```bash
python -m pip install -e ".[dev]"
pytest -q
python - <<'PY'
import agent_core
print(agent_core.__version__)
PY
```

Expected tests: `225 passed` and version `0.10.19`.

## Level-4 place pilot

```bash
EPISODES=60 WORKERS=1 \
./scripts/run_training_wsl.sh \
  --curriculum place \
  --level 4
```

Expected startup trainer: `portal_center_place_pose_curriculum`.

Watch these metrics: `door_cross`, `route_resume_target`, `place_receptacle_open`, `place_release`, setup skips, timeouts, transport failures, and restarts.
