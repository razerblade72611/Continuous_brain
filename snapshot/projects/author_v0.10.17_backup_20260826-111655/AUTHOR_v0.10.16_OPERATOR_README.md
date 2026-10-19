# AUTHOR v0.10.16 — Exact-Pose Discovery Prop Reset

v0.10.16 fixes the Level-4 setup starvation introduced by v0.10.15's Discovery-prop hygiene.

## Root cause

v0.10.15 captured an object's center position and later passed that center to AI2-THOR `PlaceObjectAtPoint`. That action is a surface-placement primitive, not an exact saved-pose restore. Once `Discovery_Book` fell from its generated desk/pedestal, restoration could fail on every subsequent raw setup.

## Fix

- Capture `Discovery_*` Unity object name, full XYZ position, and XYZ rotation at worker startup.
- Clear both agents' held objects as unscored trainer cleanup before independent episode setup.
- Detect both translation drift and in-place tipping/rotation drift.
- Restore displaced generated props with `SetObjectPoses(..., placeStationary=True)`.
- Include every current moveable/pickupable object in the `SetObjectPoses` request, preserving AI2-THOR objects that would otherwise be removed when omitted.
- Verify the restored Discovery props after the simulator call.
- Replay schema: `author.embodied.v01016`.
- SFT schema: `author.embodied_sft.v01016`.
- Existing v0.10.15 route-safe peer parking, dynamic blocker replanning, component-verified semantic routes, FIFO watchdog, transport recovery, and maneuver-clear world are preserved.

## Upgrade

Preserve the existing `state/` directory. Do not use `--reset-memory`. No Unity rebuild is required.

After installation:

```bash
pytest -q
python - <<'PY'
import agent_core
print(agent_core.__version__)
PY
```

Expected: all tests pass and version `0.10.16`.

## Validation run

```bash
EPISODES=60 WORKERS=1 \
./scripts/run_training_wsl.sh \
  --curriculum pickup \
  --level 4
```

Expected startup trainer: `exact_pose_reset_route_safe_curriculum`.

The prior v0.10.15 signature

```text
trainer could not restore displaced Discovery prop Book|-00.62|+00.86|+06.92
```

must not repeat hundreds of times. `discovery_prop_restores > 0` is valid and means trainer hygiene repaired a displaced prop before scoring the next lesson.
