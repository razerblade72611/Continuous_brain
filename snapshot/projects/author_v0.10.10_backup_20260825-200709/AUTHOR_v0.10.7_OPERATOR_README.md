# AUTHOR v0.10.7 Operator Guide

## Fix

Level-3 diagnostics exposed a v0.10.6 trainer regression. `SharedThorWorld.step_for(agent_id, ...)` returns the selected child Event. `_teleport()` and `_admin_step()` incorrectly passed that returned Event through `_selected_event(event, agent_id)` again. Agent B (`agent_id=1`) could therefore raise `IndexError: list index out of range` during peer parking or trainer admin actions.

v0.10.7 consumes the selected Event directly. The v0.10.6 simulator watchdog remains enabled.

## Upgrade

Preserve the current `state` directory when replacing v0.10.6. Do not reset A/B memory.

## Verify

```bash
pytest -q
python - <<'PY'
import agent_core
print(agent_core.__version__)
PY
```

Expected version: `0.10.7`.

## Level-3 validation

First use a 60-episode short validation:

```bash
EPISODES=60 WORKERS=1 ./scripts/run_training_wsl.sh --curriculum door --level 3
```

If it produces scored episodes and progresses normally, run the full 250-episode Level-3 curriculum. Closed-door setups with no interactable pose may still be skipped unscored by design.
