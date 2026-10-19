#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/continuous_brain_v0.12.2}"
THOR_EXECUTABLE="${THOR_EXECUTABLE:-$HOME/projects/ai2thor-humanoid/builds/thor-discovery-Linux64/thor-discovery-Linux64}"
SCENE="${SCENE:-FloorPlan1Discovery_physics}"

cd "$PROJECT_ROOT"

ROOT="state/thor_discovery_embodiment_admission_v1"
mkdir -p "$ROOT"
rm -f "$ROOT/result.json" "$ROOT/run.log" "$ROOT/result_bundle.zip"

echo "=== v0.26.0 Discovery World Embodiment Admission ==="
echo "No training. Launches the real custom AI2-THOR executable."
echo "Verifies doors, reachability, RGB/pose response, and protected cognition."
echo "scene=$SCENE"
echo "thor_executable=$THOR_EXECUTABLE"

test -x "$THOR_EXECUTABLE"

PYTHON_BIN="$(command -v python)"

if ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import ai2thor
PY
then
  if [ -x "$HOME/.venvs/author/bin/python" ] && \
     "$HOME/.venvs/author/bin/python" - <<'PY' >/dev/null 2>&1
import ai2thor
PY
  then
    PYTHON_BIN="$HOME/.venvs/author/bin/python"
    echo "AI2-THOR controller source: AUTHOR venv"
  else
    echo "ERROR: ai2thor Python controller is unavailable in both active and AUTHOR venvs."
    exit 2
  fi
else
  echo "AI2-THOR controller source: active Continuous Brain venv"
fi

"$PYTHON_BIN" -m py_compile admit_thor_discovery_embodiment_v1.py

set +e
GALLIUM_DRIVER="${GALLIUM_DRIVER:-d3d12}" \
MESA_D3D12_DEFAULT_ADAPTER_NAME="${MESA_D3D12_DEFAULT_ADAPTER_NAME:-AMD}" \
"$PYTHON_BIN" admit_thor_discovery_embodiment_v1.py \
  --scene "$SCENE" \
  --thor-executable "$THOR_EXECUTABLE" \
  --state-dir "$ROOT" \
  2>&1 | tee "$ROOT/run.log"
STATUS=${PIPESTATUS[0]}
set -e

if [ "$STATUS" -ne 0 ] && [ "$STATUS" -ne 4 ]; then
  echo "THOR_DISCOVERY_EMBODIMENT_WRAPPER: FAIL_EVALUATOR"
  exit "$STATUS"
fi

grep -q '^THOR_DISCOVERY_EMBODIMENT_ADMISSION_COMPLETE$' "$ROOT/run.log"
grep -q '^PROTECTED_UNCHANGED: true$' "$ROOT/run.log"
grep -q '^V0252_TEST_PASS_PRESERVED: true$' "$ROOT/run.log"
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

with ZipFile(b, "w", ZIP_DEFLATED) as z:
    z.write(r, "result.json")
    log = r.with_name("run.log")
    if log.is_file():
        z.write(log, "run.log")
print(b)
PY

if [ "$STATUS" -eq 0 ]; then
  echo "THOR_DISCOVERY_EMBODIMENT_ADMISSION: PASS_CONFIRMED"
else
  echo "THOR_DISCOVERY_EMBODIMENT_ADMISSION: FAIL_CONFIRMED"
fi

echo "THOR_DISCOVERY_EMBODIMENT_WRAPPER: COMPLETE"
exit "$STATUS"
