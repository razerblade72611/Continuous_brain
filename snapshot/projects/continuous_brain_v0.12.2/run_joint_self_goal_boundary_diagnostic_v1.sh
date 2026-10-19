#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/continuous_brain_v0.12.2}"
cd "$PROJECT_ROOT"

ROOT="state/joint_self_goal_boundary_diagnostic_v1"
mkdir -p "$ROOT"
rm -f "$ROOT/result.json" "$ROOT/run.log" "$ROOT/result_bundle.zip"

echo "=== v0.30.2 Joint-Self Goal Boundary Diagnostic ==="
echo "READ ONLY. No training."
echo "Reuses spent v0.30 DEV 40000-44999."
echo "Diagnoses why future-goal decoding plateaus around 97.7%."
echo "Joint TEST 60000-64999 remains sealed."

test -s state/joint_self_world_prediction_v1/result.json
test -s state/joint_self_world_prediction_v1/joint_self_world_model.pt
test -s state/joint_self_world_repair_v1/result.json
test -s state/joint_self_world_repair_v1/future_goal_repair_head.pt

python -m py_compile \
  diagnose_joint_self_goal_boundary_v1.py

python diagnose_joint_self_goal_boundary_v1.py \
  2>&1 | tee "$ROOT/run.log"

grep -q '^JOINT_SELF_GOAL_BOUNDARY_DIAGNOSTIC_COMPLETE$' "$ROOT/run.log"
grep -q '^PROTECTED_UNCHANGED: true$' "$ROOT/run.log"
grep -q '^JOINT_TEST_TOUCHED: false$' "$ROOT/run.log"
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
assert p["joint_test_touched"] is False

with ZipFile(b, "w", ZIP_DEFLATED) as z:
    z.write(r, "result.json")
    log = r.with_name("run.log")
    if log.is_file():
        z.write(log, "run.log")

print(b)
PY

echo "JOINT_SELF_GOAL_BOUNDARY_DIAGNOSTIC_WRAPPER: COMPLETE"
