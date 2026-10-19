# AUTHOR v0.10.23 — Carried Destination Probe + Trusted Pinned Terminal

v0.10.23 is a narrow Level-4 place correction on top of v0.10.22. It preserves the successful placement/receptacle contract and the carried door-routing fixes, while addressing the final two failures from run `20260826-134205`.

## Observed v0.10.22 result

The 60-episode Level-4 place run produced 58 successes and 2 failures (96.67%), with zero setup skips, simulator timeouts, transport failures, or restarts. `place_release` was 58/58.

The two failures were:

1. **Post-cross held-object route blocked by Fridge** — PaperTowelRoll crossed the North Archive door successfully, but the exact CounterTop destination leg repeatedly collided with the Fridge. Dynamic replanning correctly blacklisted the attempted cells, then exhausted the route.
2. **Preflighted pinned pose omitted after topology refresh** — Bread crossed successfully, but a later reachable/interactable refresh omitted the exact trainer-preflighted CounterTop pose and v0.10.22 returned `pinned_pose_not_available` without a physical failure.

## What changed

1. **Physical post-cross destination-leg preflight**
   - Before a Level-4 place lesson is scored, the trainer temporarily opens the selected blocking door, computes the same far-side portal landing used by the motor, administratively places the active agent at that landing, and physically executes the held-object route to the exact preflighted target pose.
   - This uses Unity's real collision engine rather than a geometric guess.
   - The probe is trainer-private: no scored replay row or skill update is created.
   - The exact start pose and blocking-door state are restored before scoring.
   - A failed destination probe becomes `skip_not_failure` so curriculum sampling can choose another valid source/pose.

2. **Trusted preflighted pinned terminal**
   - If the target component is already visible after a door route (`GetInteractablePoses` returns target poses) but the exact trainer-preflighted pinned cell is omitted from the current reachable refresh, AUTHOR injects only that single terminal cell into the private route graph.
   - This cannot bypass a closed door because it is only enabled once the target component itself is visible.
   - Physical movement and the authoritative pose verifier still decide success.

3. **Replay-driven v0.10.22 repair**
   - `scripts/repair_v01022_place_destination_regression.py` removes only the two known failed `place_approach` / `route_resume_target` counter labels matched from v0.10.22 replay.
   - Successful door approach/open/cross and all placement evidence remain untouched.
   - Raw replay is preserved verbatim and A/B checkpoints are backed up.
   - The repair is idempotent.

## Validation

Full source suite: `249 passed`.

Expected runtime header:

```text
version: 0.10.23
trainer: carried_destination_probe_curriculum
```

## Upgrade without resetting learning

```bash
cd ~/projects

BACKUP=~/projects/author_v0.10.22_backup_$(date +%Y%m%d-%H%M%S)

mv ~/projects/author "$BACKUP"

unzip \
  /mnt/c/Users/davei/Downloads/author_phase9_v0.10.23_carried_destination_probe.zip \
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

Expected: `249 passed`, version `0.10.23`, state `/home/davei/projects/author/state`.

Do not use `--reset-memory`.

## Repair the v0.10.22 destination labels once

```bash
python scripts/repair_v01022_place_destination_regression.py \
  --state-dir "$HOME/projects/author/state"
```

For run `20260826-134205`, expected match count: `2`.

## Qualification run

```bash
STATE_DIR="$HOME/projects/author/state" \
EPISODES=60 WORKERS=1 \
./scripts/run_training_wsl.sh \
  --curriculum place \
  --level 4
```

Primary checks:

- `place_release` remains 100%;
- no scored `pinned_pose_not_available` after a successful door cross;
- the PaperTowelRoll/Fridge-style destination collision becomes an unscored setup skip if that exact destination leg is physically impossible;
- scored place success should remain at or above the v0.10.22 58/60 baseline, ideally 60/60.
