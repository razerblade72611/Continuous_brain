# AUTHOR v0.10.22 — Carried Route Admission + Primitive-True Replanning

v0.10.22 is a narrow Level-4 place routing correction on top of v0.10.21. It preserves the stable v0.10.19+ placement/receptacle contract and fixes the v0.10.21 regression exposed by run `20260826-131316`.

## Observed v0.10.21 failure classes

The 60-episode Level-4 place run produced 38 successes and 22 failures. Replay showed:

- 17 zero-step scored `place_approach` failures with no simulator error and no door auxiliary stages. These were synthetic admission failures: the carried door selector compared candidate yaw to a body-only shortest-path arrival yaw, even though the carried planner can strafe/back while preserving orientation.
- 2 `blocking_door_approach_failed` collisions against the closed North Archive door while carrying an object.
- 2 `door_cross` rows with `verification_basis=cross_primitive_exception`.
- 1 post-cross `pinned_pose_replans_exhausted` failure where a physical blocker was detected, but v0.10.21 blacklisted a cell inferred from the unrelated body-only path instead of the cell attempted by the failed carried primitive.
- 0 `place_release` failures. The PutObject/receptacle contract remains unchanged.

## What changed

1. **Planner-consistent carried door admission**
   - Door interaction-pose admission now calls `_carried_safe_pose_actions` directly.
   - The old body-only `arrival_yaw` gate is removed.
   - Carried door poses must still face the architectural doorway center and remain laterally centered.

2. **Held-object door stand-off**
   - Carried door-opening poses must remain far enough from the closed panel for the held object's horizontal AABB footprint.
   - This prevents the body from entering a body-valid pose while the held object intersects the still-closed door.

3. **Pre-score Level-4 route admission**
   - After the trainer prepares the held source and parks the passive peer, the carried route is statically admitted before `place_approach` is scored.
   - Motor-contract-incompatible setups become `skip_not_failure` and are retried by the curriculum.

4. **Primitive-true dynamic obstacle replanning**
   - For failed MoveAhead/MoveBack/MoveLeft/MoveRight actions, the blocked grid cell is derived from the actual failed primitive and current yaw.
   - Body-only shortest-path inference remains only as a fallback.

5. **Carried portal staging uses the carried planner**
   - CrossThreshold reaches the source-side staging pose with `_carried_safe_pose_actions` when holding an object.
   - The staging goal includes the required portal yaw, so the motor turns in a clearance-qualified cell before the straight crossing.

6. **Replay-driven v0.10.21 repair**
   - `scripts/repair_v01021_place_route_admission_regression.py` removes only replay-matched failed skill-counter evidence for the known v0.10.21 route-admission/replan regressions.
   - Successful door/open/cross/place evidence is preserved.
   - Raw replay is preserved verbatim.
   - A/B checkpoints are backed up before mutation.
   - The repair is idempotent.

## Validation

Source suite: `244 passed`.

Expected runtime header:

```text
version: 0.10.22
trainer: carried_route_admission_curriculum
```

## Upgrade without resetting learning

```bash
cd ~/projects

BACKUP=~/projects/author_v0.10.21_backup_$(date +%Y%m%d-%H%M%S)

mv ~/projects/author "$BACKUP"

unzip \
  /mnt/c/Users/davei/Downloads/author_phase9_v0.10.22_carried_route_admission.zip \
  -d ~/projects

rm -rf ~/projects/author/state
cp -a "$BACKUP/state" ~/projects/author/state

cd ~/projects/author
source ~/.venvs/author/bin/activate
python -m pip install -e ".[dev]"
pytest -q

python - <<'PY'
import agent_core
print(agent_core.__version__)
PY

readlink -f state
```

Expected: `244 passed`, version `0.10.22`, state `/home/davei/projects/author/state`.

Do not use `--reset-memory`.

## Repair the v0.10.21 regression labels once

```bash
python scripts/repair_v01021_place_route_admission_regression.py \
  --state-dir "$HOME/projects/author/state"
```

For run `20260826-131316`, the expected match count is 22.

## Qualification run

```bash
STATE_DIR="$HOME/projects/author/state" \
EPISODES=60 WORKERS=1 \
./scripts/run_training_wsl.sh \
  --curriculum place \
  --level 4
```

Primary checks:

- no zero-step scored `place_approach` rows with no reason/error;
- `place_release` remains 100%;
- `blocking_door_approach_failed` collapses;
- `cross_primitive_exception` collapses;
- post-cross blocker telemetry shows `dynamic_obstacle_replan` with `failed_primitive` and the attempted blocked cell.
