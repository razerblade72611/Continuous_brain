# AUTHOR v0.10.25 — Peer-Neutral Toggle Route Qualification

v0.10.25 fixes the Level-4 toggle setup-search starvation observed in v0.10.24 after 37/37 successful scored episodes. The preserved worker replay proved the valid toggle pool was broad (StoveKnob, LightSwitch, CoffeeMachine, DeskLamp, Microwave, CellPhone, Faucet, Toaster), so starvation was not caused by one broken target. The trainer bug was stateful multi-agent setup contamination: route qualification ran before the passive agent was re-parked, and skipped setups keep the same scored episode/active agent. The passive agent could therefore remain indefinitely at the previous episode's terminal pose and poison GetReachablePositions/GetInteractablePoses during every retry.

## Changes

- Before Level-4 toggle blocked-route discovery, open all doors and park the passive agent in a neutral maneuverable cell away from doors, the active agent, and the current toggle target pool.
- After a route tuple is selected, retain the existing route-specific `_park_peer_off_level4_route` placement for the scored lesson.
- Keep Level 5 peer interference semantics unchanged.
- Preserve the v0.10.24 `isBroken` private trainer/runtime guard; healthy Laptops remain eligible while physically broken targets are never scored for impossible ToggleObjectOn actions.
- No changes to pickup, place, door crossing, placement release, Unity geometry, or the semantic planner.
- Replay schema: `author.embodied.v01025`.
- SFT schema: `author.embodied_sft.v01025`.
- Full regression suite: 261 passed.

## Important state note

The interrupted v0.10.24 worker containing 37/37 successful toggle episodes is diagnostic evidence only. Do not manually merge it. Copying the active `state` directory during this upgrade preserves the archived worker evidence without changing the authoritative merged A/B checkpoints.

## Upgrade from v0.10.24

```bash
cd ~/projects
BACKUP=~/projects/author_v0.10.24_backup_$(date +%Y%m%d-%H%M%S)
mv ~/projects/author "$BACKUP"
unzip /mnt/c/Users/davei/Downloads/author_phase9_v0.10.25_peer_neutral_toggle_route.zip -d ~/projects
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

Expected: `261 passed`, version `0.10.25`, active state `/home/davei/projects/author/state`.

No replay repair is required for the interrupted v0.10.24 starvation run because those rejected setups were never scored or merged.

## Final Level-4 toggle qualification

```bash
STATE_DIR="$HOME/projects/author/state" \
EPISODES=60 WORKERS=1 \
./scripts/run_training_wsl.sh \
  --curriculum toggle \
  --level 4
```

Expected header: version `0.10.25`, trainer `peer_neutral_toggle_route_curriculum`, state `/home/davei/projects/author/state`.

Healthy setup skips remain acceptable, but the pathological pattern `next episode 38/60` with hundreds of repeated `no functional toggleable target...` rejections should not recur. Do not use `--reset-memory`. No Unity rebuild is required.
