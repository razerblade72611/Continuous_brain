# AUTHOR v0.10.17 — Canonical Scene Reset

## Why this release exists
The v0.10.16 Level-4 pickup run scored 30 clean episodes, then episode 31 accumulated hundreds of raw setup rejections while Unity and the trainer remained active. Source inspection found that successful pickup still used `PlaceObjectAtPoint` to restore the picked target. That is a surface-placement primitive, not an exact saved-transform restore. Only `Discovery_*` props were covered by the newer exact-pose baseline, so stock pickup targets could drift between lessons.

## Fix
- The trainer captures the canonical full pose (position + rotation) of every movable/pickupable object at worker start.
- Before every independent curriculum attempt it clears both hands, detects displaced/missing baseline objects, and restores the full movable scene with `SetObjectPoses(placeStationary=True)`.
- Successful pickup no longer calls `PlaceObjectAtPoint`; it only clears the hand. The next lesson's canonical reset restores the scene exactly.
- Setup-search status is printed every 25 unscored setup rejections.
- No Unity rebuild is required.
- Never use `--reset-memory` for this upgrade.

## Upgrade
```bash
cd ~/projects
BACKUP=~/projects/author_v0.10.16_backup_$(date +%Y%m%d-%H%M%S)
mv ~/projects/author "$BACKUP"
unzip /mnt/c/Users/davei/Downloads/author_phase9_v0.10.17_canonical_scene_reset.zip -d ~/projects
cp -a "$BACKUP/state" ~/projects/author/
cd ~/projects/author
source ~/.venvs/author/bin/activate
python -m pip install -e ".[dev]"
pytest -q
python - <<'PY'
import agent_core
print(agent_core.__version__)
PY
```
Expected version: `0.10.17`.

## Level-4 validation
```bash
EPISODES=60 WORKERS=1 \
./scripts/run_training_wsl.sh \
  --curriculum pickup \
  --level 4
```
Startup trainer: `canonical_scene_reset_semantic_route_curriculum`.

If setup rejection repeats, v0.10.17 prints a status line every 25 skips with the latest reason rather than appearing frozen.
