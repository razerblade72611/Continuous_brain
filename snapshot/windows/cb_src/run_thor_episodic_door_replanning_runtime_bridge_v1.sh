#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/continuous_brain_v0.12.2}"
AUTHOR_PYTHON="${AUTHOR_PYTHON:-$HOME/.venvs/author/bin/python}"
THOR_EXECUTABLE="${THOR_EXECUTABLE:-$HOME/projects/ai2thor-humanoid/builds/thor-discovery-Linux64/thor-discovery-Linux64}"
SCENE="${SCENE:-FloorPlan1Discovery_physics}"

cd "$PROJECT_ROOT"

ROOT="state/thor_episodic_door_replanning_v1"
mkdir -p "$ROOT"
rm -f "$ROOT/result.json" "$ROOT/run.log" "$ROOT/result_bundle.zip"

echo "=== v0.27.1 Runtime Bridge Repair for v0.27.0 ==="
echo "No training. No evaluator changes. No split changes."
echo "Uses Continuous Brain Python + AUTHOR ai2thor fallback path."
echo "scene=$SCENE"
echo "thor_executable=$THOR_EXECUTABLE"

test -x "$THOR_EXECUTABLE"
test -x "$AUTHOR_PYTHON"
test -s evaluate_thor_episodic_door_replanning_v1.py
test -s continuous_brain/thor_episodic_door_replanning_v1.py

CURRENT_PYTHON="$(command -v python)"

echo "Continuous Brain Python: $CURRENT_PYTHON"
echo "AUTHOR Python:           $AUTHOR_PYTHON"

"$CURRENT_PYTHON" - <<'PY'
import torch
print("Continuous Brain torch:", torch.__version__)
print("Continuous Brain torch file:", torch.__file__)
PY

AUTHOR_SITE="$("$AUTHOR_PYTHON" - <<'PY'
import site
paths = site.getsitepackages()
if not paths:
    raise SystemExit("AUTHOR venv has no site-packages path")
print(paths[0])
PY
)"

echo "AUTHOR site-packages: $AUTHOR_SITE"

"$AUTHOR_PYTHON" - <<'PY'
import ai2thor
from ai2thor.controller import Controller
print("AUTHOR ai2thor:", ai2thor.__file__)
PY

"$CURRENT_PYTHON" -m py_compile \
  run_with_author_ai2thor_bridge_v1.py \
  continuous_brain/thor_episodic_door_replanning_v1.py \
  evaluate_thor_episodic_door_replanning_v1.py \
  test_thor_episodic_door_replanning_v1.py

"$CURRENT_PYTHON" test_thor_episodic_door_replanning_v1.py

# Verify the merged import environment before launching Unity.
"$CURRENT_PYTHON" run_with_author_ai2thor_bridge_v1.py \
  --author-site "$AUTHOR_SITE" \
  --script /dev/null \
  2>&1 | sed -n \
  '/RUNTIME_BRIDGE_IMPORT:/p;/current_python:/p;/torch_version:/p;/torch_file:/p;/ai2thor_file:/p;/author_site_appended_last:/p'

# Run the UNCHANGED v0.27 evaluator through the merged import environment.
set +e
GALLIUM_DRIVER="${GALLIUM_DRIVER:-d3d12}" \
MESA_D3D12_DEFAULT_ADAPTER_NAME="${MESA_D3D12_DEFAULT_ADAPTER_NAME:-AMD}" \
"$CURRENT_PYTHON" run_with_author_ai2thor_bridge_v1.py \
  --author-site "$AUTHOR_SITE" \
  --script "$PROJECT_ROOT/evaluate_thor_episodic_door_replanning_v1.py" \
  -- \
  --scene "$SCENE" \
  --thor-executable "$THOR_EXECUTABLE" \
  --state-dir "$ROOT" \
  2>&1 | tee "$ROOT/run.log"
STATUS=${PIPESTATUS[0]}
set -e

if [ "$STATUS" -ne 0 ] && [ "$STATUS" -ne 4 ]; then
  echo "THOR_EPISODIC_DOOR_REPLANNING_RUNTIME_BRIDGE: FAIL_EVALUATOR"
  exit "$STATUS"
fi

grep -q '^RUNTIME_BRIDGE_IMPORT: PASS$' "$ROOT/run.log"
grep -q '^THOR_EPISODIC_DOOR_REPLANNING_RUN_COMPLETE$' "$ROOT/run.log"
grep -q '^PROTECTED_UNCHANGED: true$' "$ROOT/run.log"
grep -q '^V0252_TEST_PASS_PRESERVED: true$' "$ROOT/run.log"
grep -q '^V026_ADMISSION_PASS_PRESERVED: true$' "$ROOT/run.log"
test -s "$ROOT/result.json"

"$CURRENT_PYTHON" - "$ROOT/result.json" "$ROOT/result_bundle.zip" <<'PY'
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

echo "THOR_EPISODIC_DOOR_REPLANNING_RUNTIME_BRIDGE: COMPLETE"
exit "$STATUS"
