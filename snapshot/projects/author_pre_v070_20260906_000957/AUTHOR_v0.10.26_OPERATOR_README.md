# AUTHOR v0.10.26 — ExitRegion Motor Handoff

v0.10.26 fixes the live runtime failure exposed by the task `Leave the kitchen.` The task parser and simulator verifier already recognized a connected-region exit, but the motor handoff required the LLM to first ground a visible Door focus. With the local Qwen2-VL model this produced long runs of raw `MoveAhead`/`Observe` decisions and repeated `GroundingError` messages instead of invoking the trained door skills.

## Runtime contract

For an explicit Discovery-world `ExitRegion` task:

1. The semantic layer receives only the user-authorized outcome: physically leave the starting region.
2. If the model has already grounded a Door, AUTHOR preserves that choice.
3. Otherwise the private motor layer selects the nearest physical Door using hidden simulator metadata; no object ID or coordinate is exposed to the LLM.
4. If needed, `NavigateTo` uses the trained `door_approach` skill.
5. The motor opens the selected door with `door_open`.
6. The motor crosses with `CrossThreshold` / `door_cross`.
7. The existing `ExitRegion` simulator verifier completes the task only after the body is physically in another Discovery region.

Primitive LLM movement remains available for general autonomy, but during a recognized explicit ExitRegion task it is preparatory input and may be replaced by the verified motor handoff.

Stock/non-Discovery scenes are unchanged because AUTHOR intentionally has no region-boundary verifier there.

## Preserved behavior

- v0.10.25 peer-neutral Level-4 toggle setup qualification remains intact.
- v0.10.24 broken-toggle target/runtime affordance guard remains intact.
- Level-4 pickup/place/toggle motor semantics are unchanged.
- No Unity rebuild is required.
- Do not use `--reset-memory`.

## Upgrade from v0.10.25

```bash
cd ~/projects
BACKUP=~/projects/author_v0.10.25_backup_$(date +%Y%m%d-%H%M%S)
mv ~/projects/author "$BACKUP"
unzip /mnt/c/Users/davei/Downloads/author_phase9_v0.10.26_exit_region_motor_handoff.zip -d ~/projects
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

## Single-agent runtime validation

```bash
cd ~/projects/author
source ~/.venvs/author/bin/activate

WINDOWS_HOST="$(ip route show | awk '/default/ {print $3; exit}')"
export AGENT_LLM_BASE_URL="http://${WINDOWS_HOST}:8080/v1"
export AGENT_LLM_MODEL="qwen2-vl-2b"

GALLIUM_DRIVER=d3d12 \
MESA_D3D12_DEFAULT_ADAPTER_NAME=AMD \
python -m agent_core.cli \
  --scene FloorPlan1Discovery_physics \
  --session \
  --llm \
  --vision \
  --interact \
  --initial-task "Leave the kitchen." \
  --thor-executable "$HOME/projects/ai2thor-humanoid/builds/thor-discovery-Linux64/thor-discovery-Linux64" \
  --state-dir "$HOME/projects/author/state" \
  --vision-every 1 \
  --roam-decisions 100 \
  --max-steps 600 \
  --step-delay 0.08
```

Expected behavior is a short staged sequence dominated by `Approach`/`OpenObject`/`CrossThreshold` motor handoffs followed by `[task-complete] Leave the kitchen.` It should no longer spend dozens of decisions repeatedly issuing raw `MoveAhead` toward a visually described door.
