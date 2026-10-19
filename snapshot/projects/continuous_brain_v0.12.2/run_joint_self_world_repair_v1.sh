#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/continuous_brain_v0.12.2}"
cd "$PROJECT_ROOT"

ROOT="state/joint_self_world_repair_v1"
mkdir -p "$ROOT"
rm -f \
  "$ROOT/result.json" \
  "$ROOT/run.log" \
  "$ROOT/result_bundle.zip" \
  "$ROOT/future_goal_repair_head.pt" \
  "$ROOT/joint_self_reload_probe.pt"

echo "=== v0.30.1 Joint Self/World DEV Repair ==="
echo "Source v0.30.0 remains formal FAIL."
echo "Freezes the joint self/world model."
echo "Repairs only future-goal decoding and the invalid counterfactual control."
echo "Reuses DEV 40000-44999. TEST 60000-64999 remains sealed."

test -s state/joint_self_world_prediction_v1/result.json
test -s state/joint_self_world_prediction_v1/joint_self_world_model.pt
test -s state/self_broadcast_centrality_test_v1/result.json

python -m py_compile \
  continuous_brain/joint_self_world_repair_v1.py \
  train_evaluate_joint_self_world_repair_v1.py \
  test_joint_self_world_repair_v1.py

python test_joint_self_world_repair_v1.py

set +e
python train_evaluate_joint_self_world_repair_v1.py \
  --state-dir "$ROOT" \
  2>&1 | tee "$ROOT/run.log"
STATUS=${PIPESTATUS[0]}
set -e

if [ "$STATUS" -ne 0 ] && [ "$STATUS" -ne 4 ]; then
  echo "JOINT_SELF_WORLD_REPAIR_WRAPPER: FAIL_EVALUATOR"
  exit "$STATUS"
fi

grep -q '^JOINT_SELF_WORLD_REPAIR_COMPLETE$' "$ROOT/run.log"
grep -q '^SOURCE_MODEL_UNCHANGED: true$' "$ROOT/run.log"
grep -q '^PROTECTED_UNCHANGED: true$' "$ROOT/run.log"
grep -q '^JOINT_TEST_TOUCHED: false$' "$ROOT/run.log"

test -s "$ROOT/result.json"
test -s "$ROOT/future_goal_repair_head.pt"

python - \
  "$ROOT/result.json" \
  "$ROOT/future_goal_repair_head.pt" \
  "$ROOT/result_bundle.zip" <<'PY'
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import json, sys

r=Path(sys.argv[1])
h=Path(sys.argv[2])
b=Path(sys.argv[3])
p=json.loads(r.read_text())

assert p["source_v0300_formal_status"] == "FAIL"
assert p["source_model_unchanged"] is True
assert p["protected_unchanged"] is True
assert p["joint_test_touched"] is False
assert p["repair_scope"]["joint_model_training"] is False
assert p["repair_scope"]["world_dynamics_training"] is False
assert p["repair_scope"]["self_dynamics_training"] is False

with ZipFile(b,"w",ZIP_DEFLATED) as z:
    z.write(r,"result.json")
    z.write(h,"future_goal_repair_head.pt")
    log=r.with_name("run.log")
    if log.is_file():
        z.write(log,"run.log")

print(b)
PY

if [ "$STATUS" -eq 0 ]; then
  echo "JOINT_SELF_WORLD_REPAIR_DEV_GATE: PASS_CONFIRMED"
else
  echo "JOINT_SELF_WORLD_REPAIR_DEV_GATE: FAIL_CONFIRMED"
fi

echo "JOINT_SELF_WORLD_REPAIR_WRAPPER: COMPLETE"
exit "$STATUS"
