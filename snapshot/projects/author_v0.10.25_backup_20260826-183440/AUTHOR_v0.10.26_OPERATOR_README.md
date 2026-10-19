# AUTHOR v0.10.26 — Peer-Authoritative Neutral Parking

v0.10.26 fixes the second Level-4 toggle setup-starvation failure observed in v0.10.25 after 57/57 successful scored episodes. The motor and toggle interaction were clean; the new pre-search peer-neutral helper itself became the rejection point.

## Root cause addressed

v0.10.25 selected candidate neutral cells from the active agent's `GetReachablePositions()` result and then attempted to `TeleportFull` the passive peer into those cells. In a visible multi-agent scene, reachability is agent-specific because the other agent is a live collider. v0.10.26 treats the passive peer's own reachability as authoritative for passive-peer placement.

## Changes

- Query both active and passive agents for reachability before Level-4 toggle route discovery.
- Prefer neutral cells present in both reachability views; fall back to peer-authoritative cells when necessary.
- If the passive peer is already in a neutral maneuverable position, accept it without forcing a redundant teleport.
- If direct passive-peer parking fails, perform a trainer-only two-agent neutral normalization: move the active body to one safe cell and the peer to another far-separated safe cell, then let normal blocked-route qualification place the active agent at its scored start.
- Setup failures now include explicit counts for active/peer reachable cells, common cells, neutral candidates, failed peer teleports, and pair-normalization attempts.
- Preserve v0.10.24 broken-toggle admission and all v0.10.23 carried-object/place contracts.
- No Unity rebuild, no semantic planner changes, no pickup/place/door motor changes.
- Replay schema: `author.embodied.v01026`.
- SFT schema: `author.embodied_sft.v01026`.

## State handling

The interrupted v0.10.25 worker is diagnostic evidence only. Do not manually merge it. Copy the active `state` directory during upgrade; this preserves merged A/B learning and any archived worker diagnostics.

## Upgrade

```bash
cd ~/projects
BACKUP=~/projects/author_v0.10.25_backup_$(date +%Y%m%d-%H%M%S)
mv ~/projects/author "$BACKUP"
unzip /mnt/c/Users/davei/Downloads/author_phase9_v0.10.26_peer_authoritative_neutral.zip -d ~/projects
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

Do not use `--reset-memory`. No replay repair is required for the interrupted v0.10.25 starvation run.

## Level-4 toggle qualification

```bash
STATE_DIR="$HOME/projects/author/state" \
EPISODES=60 WORKERS=1 \
./scripts/run_training_wsl.sh \
  --curriculum toggle \
  --level 4
```

Expected header: version `0.10.26`, trainer `peer_authoritative_neutral_curriculum`, active state `/home/davei/projects/author/state`.
