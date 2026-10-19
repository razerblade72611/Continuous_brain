#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/continuous_brain_v0.12.2}"
THOR_EXECUTABLE="${THOR_EXECUTABLE:-$HOME/projects/ai2thor-humanoid/builds/thor-discovery-Linux64/thor-discovery-Linux64}"
SCENE="${SCENE:-FloorPlan1Discovery_physics}"

cd "$PROJECT_ROOT"

ROOT="state/thor_episodic_door_replanning_v1"
mkdir -p "$ROOT"
rm -f "$ROOT/result.json" "$ROOT/run.log" "$ROOT/result_bundle.zip"

echo "=== v0.27.0 Real 3-D Episodic Door Replanning Bridge ==="
echo "No cognitive training."
echo "Uses real Discovery door identities/positions as bound episodic payloads."
echo "FIRST/SECOND select a real door goal through frozen v0.23 memory attention."
echo "Physically navigates, opens, crosses, detects surprise close, and replans."
echo "scene=$SCENE"
echo "thor_executable=$THOR_EXECUTABLE"

test -x "$THOR_EXECUTABLE"
test -s state/consequence_replanning_test_v1/result.json
test -s state/thor_discovery_embodiment_admission_v1/result.json
test -s state/multi_item_episodic_memory_v1/episodic_reader.pt
test -s state/working_memory_recall_capacity_control_phase1_v1/latest.portable.pt

python -m py_compile \
  continuous_brain/thor_episodic_door_replanning_v1.py \
  evaluate_thor_episodic_door_replanning_v1.py \
  test_thor_episodic_door_replanning_v1.py

python test_thor_episodic_door_replanning_v1.py

PYTHON_BIN="$(command -v python)"

if ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import ai2thor
PY
then
  if [ -x "$HOME/.venvs/author/bin/python" ] && \
     "$HOME/.venvs/author/bin/python" - <<'PY' >/dev/null 2>&1
import ai2thor
import torch
PY
  then
    PYTHON_BIN="$HOME/.venvs/author/bin/python"
    echo "AI2-THOR controller source: AUTHOR venv"
  else
    echo "ERROR: no usable Python environment with ai2thor + torch."
    exit 2
  fi
else
  echo "AI2-THOR controller source: active Continuous Brain venv"
fi

# Force this project's source tree to win even when AUTHOR's Python provides
# the simulator package.
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_BIN" - <<'PY'
import continuous_brain
from continuous_brain.memory_guided_goal_option_v1 import build_substrate
from continuous_brain.multi_item_episodic_memory_v1 import load_reader
print("3-D bridge cognitive imports: PASS")
PY

set +e
GALLIUM_DRIVER="${GALLIUM_DRIVER:-d3d12}" \
MESA_D3D12_DEFAULT_ADAPTER_NAME="${MESA_D3D12_DEFAULT_ADAPTER_NAME:-AMD}" \
"$PYTHON_BIN" evaluate_thor_episodic_door_replanning_v1.py \
  --scene "$SCENE" \
  --thor-executable "$THOR_EXECUTABLE" \
  --state-dir "$ROOT" \
  2>&1 | tee "$ROOT/run.log"
STATUS=${PIPESTATUS[0]}
set -e

if [ "$STATUS" -ne 0 ] && [ "$STATUS" -ne 4 ]; then
  echo "THOR_EPISODIC_DOOR_REPLANNING_WRAPPER: FAIL_EVALUATOR"
  exit "$STATUS"
fi

grep -q '^THOR_EPISODIC_DOOR_REPLANNING_RUN_COMPLETE$' "$ROOT/run.log"
grep -q '^PROTECTED_UNCHANGED: true$' "$ROOT/run.log"
grep -q '^V0252_TEST_PASS_PRESERVED: true$' "$ROOT/run.log"
grep -q '^V026_ADMISSION_PASS_PRESERVED: true$' "$ROOT/run.log"
test -s "$ROOT/result.json"

python - "$ROOT/result.json" "$ROOT/result_bundle.zip" <<'PY'
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import json, sys

r = Path(sys.argv[1])
b = Path(sys.argv[2])
p = json.loads(r.read_text())

assert p["training_performed"] is False
assert p["protected_unchanged"] is True
assert p["v0252_test_pass_preserved"] is True
assert p["v026_admission_pass_preserved"] is True
assert p["scope"]["raw_visual_memory_write_claimed"] is False
assert p["scope"]["learned_3d_consequence_model_claimed"] is False

with ZipFile(b, "w", ZIP_DEFLATED) as z:
    z.write(r, "result.json")
    log = r.with_name("run.log")
    if log.is_file():
        z.write(log, "run.log")

print(b)
PY

if [ "$STATUS" -eq 0 ]; then
  echo "THOR_EPISODIC_DOOR_REPLANNING_DEV_GATE: PASS_CONFIRMED"
else
  echo "THOR_EPISODIC_DOOR_REPLANNING_DEV_GATE: FAIL_CONFIRMED"
fi

echo "THOR_EPISODIC_DOOR_REPLANNING_WRAPPER: COMPLETE"
exit "$STATUS"
