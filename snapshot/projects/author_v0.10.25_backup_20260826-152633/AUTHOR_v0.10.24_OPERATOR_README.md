# AUTHOR v0.10.24 — Broken Toggle Target Admission

v0.10.24 is a narrow Level-4 toggle correction on top of v0.10.23. The preceding run reached every toggle target (60/60 toggle_approach) but four ToggleObjectOn attempts failed on the same AI2-THOR Laptop because its metadata reported a physically broken state.

## Changes

- Filters `isBroken=True` toggleable objects before route qualification/scoring.
- Keeps healthy Laptops and all other functional toggleable targets eligible.
- Applies the same private broken-state guard to normal runtime grounding, so `ToggleObjectOn` is not advertised or grounded for a broken focus.
- Leaves door routing, place, pickup, interaction-pose recovery, and Unity world geometry unchanged.
- Adds an idempotent v0.10.23 repair that removes only the four replay-matched broken-Laptop `toggle_interaction` failures while preserving 60/60 successful `toggle_approach` evidence and raw replay.
- Replay schema: `author.embodied.v01024`.
- SFT schema: `author.embodied_sft.v01024`.

## Upgrade

```bash
cd ~/projects
BACKUP=~/projects/author_v0.10.23_backup_$(date +%Y%m%d-%H%M%S)
mv ~/projects/author "$BACKUP"
unzip /mnt/c/Users/davei/Downloads/author_phase9_v0.10.24_broken_toggle_admission.zip -d ~/projects
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

## Repair v0.10.23 invalid toggle labels

```bash
python scripts/repair_v01023_broken_toggle_target_regression.py --state-dir "$HOME/projects/author/state"
```

Expected on run `20260826-142312`: `matched_broken_toggle_regressions: 4`, raw replay preserved.

## Final Level-4 toggle qualification

```bash
STATE_DIR="$HOME/projects/author/state" EPISODES=60 WORKERS=1 ./scripts/run_training_wsl.sh --curriculum toggle --level 4
```

Expected header: version `0.10.24`, trainer `toggle_broken_target_admission_curriculum`, state `/home/davei/projects/author/state`.

Do not use `--reset-memory`. No Unity rebuild is required.
