# AUTHOR v0.10.21 — Carried-Orientation Route Clearance

v0.10.21 is a narrow Level-4 place transport correction on top of v0.10.20. It preserves the now-stable place/receptacle contract and fixes the remaining carried-object route failures seen in run `20260826-124340`.

## Evidence from v0.10.20

The completed Level-4 place batch reached 44/60 (73.33%) with zero setup skips, simulator timeouts, transport failures, or restarts. Replay classification showed:

- 44/44 `place_release` successes and zero release failures.
- 16 `place_approach` failures.
- 10 failures after a successful door cross while resuming the pinned receptacle pose.
- 5 failures before door open while approaching a door pose near a portal jamb.
- 1 true door-cross failure.
- 8 held-item 90-degree rotation collision errors.
- 5 explicit portal-boundary collisions (4 east, 1 north).
- 1 physical Fridge blocker whose Unity wording (`blocking the Agent`) did not match the old dynamic-replan string.
- 1 preflighted pinned pose omitted by a later `GetInteractablePoses` query.

Final `PutObject`, openable-receptacle handling, canonical scene reset, semantic target preservation, and pose-qualified place preflight are intentionally unchanged.

## Motor changes

- Add orientation-aware carried-object routing to exact interaction poses.
- While carrying, body rotations are allowed only in locally clearance-qualified reachable cells.
- The route may use `MoveAhead`, `MoveBack`, `MoveLeft`, and `MoveRight` to preserve body yaw through tight corridors instead of rotating at arbitrary shortest-path corners.
- The route goal includes the exact pinned interaction-pose yaw, so the agent can turn in a clear bay and translate into the target already aligned.
- Carried-object closed-door interaction poses must be centered laterally in the actual doorway opening; unsafe jamb-side generic fallback is disabled.
- Dynamic blocker recognition now accepts Unity's real `"is blocking the Agent"` wording, not only the historical `"is blocking Agent"` variant.
- A trainer-private place pose that was physically preflighted can be reused if its grid cell remains reachable after a topology change even when a later `GetInteractablePoses` query omits that exact pose. Simulator-authoritative pose verification still decides success.
- If no clearance-qualified carried route exists, the motor fails closed rather than silently reverting to the old body-only path.

## State repair

The completed v0.10.20 run merged 16 replay-proven carried-route implementation failures into persistent A/B counters. After installing v0.10.21, run once:

```bash
python scripts/repair_v01020_place_carry_route_regression.py \
  --state-dir "$HOME/projects/author/state"
```

The repair is idempotent. It backs up A/B checkpoints, subtracts only matching failed v0.10.20 Level-4 place approach evidence and failed auxiliary route stages, and leaves raw replay unchanged. Successful route/open/cross/place evidence is preserved.

For run `20260826-124340`, expected matched episodes: `16`.

## Upgrade

Preserve existing learning state. Do **not** use `--reset-memory`.

```bash
cd ~/projects
BACKUP=~/projects/author_v0.10.20_backup_$(date +%Y%m%d-%H%M%S)
mv ~/projects/author "$BACKUP"
unzip /mnt/c/Users/davei/Downloads/author_phase9_v0.10.21_carried_orientation_clearance.zip -d ~/projects
cp -a "$BACKUP/state" ~/projects/author/state
cd ~/projects/author
source ~/.venvs/author/bin/activate
python -m pip install -e ".[dev]"
pytest -q
```

Expected: `238 passed`, version `0.10.21`.

Repair the v0.10.20 regression labels:

```bash
python scripts/repair_v01020_place_carry_route_regression.py \
  --state-dir "$HOME/projects/author/state"
```

Then run the next Level-4 place qualification with an explicit absolute state path:

```bash
cd ~/projects/author
source ~/.venvs/author/bin/activate

STATE_DIR="$HOME/projects/author/state" \
EPISODES=60 WORKERS=1 \
./scripts/run_training_wsl.sh \
  --curriculum place \
  --level 4
```

Expected startup trainer: `carried_orientation_clearance_curriculum`.

No Unity rebuild is required.
