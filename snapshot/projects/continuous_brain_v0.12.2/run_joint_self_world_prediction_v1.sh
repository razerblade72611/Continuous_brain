#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/continuous_brain_v0.12.2}"
cd "$PROJECT_ROOT"

ROOT="state/joint_self_world_prediction_v1"
mkdir -p "$ROOT"
rm -f \
  "$ROOT/result.json" \
  "$ROOT/run.log" \
  "$ROOT/result_bundle.zip" \
  "$ROOT/joint_self_world_model.pt" \
  "$ROOT/joint_self_reload_probe.pt"

echo "=== v0.30.0 Joint Self / World Prediction ==="
echo "Cognition milestone. No robot training."
echo "New development family after the closed v0.29 TEST."
echo "Predicts both future external state and future persistent-self state."
echo "Tests whether self-consequences change action choice when world outcome is held fixed."
echo "Joint TEST 60000-64999 remains sealed."

test -s state/persistent_self_autobiographical_v1/agency_self_model.pt
test -s state/persistent_self_autobiographical_v1/persistent_self_seed.pt
test -s state/persistent_self_autobiographical_test_v1/result.json
test -s state/self_broadcast_centrality_test_v1/result.json

python -m py_compile \
  continuous_brain/joint_self_world_prediction_v1.py \
  train_evaluate_joint_self_world_prediction_v1.py \
  test_joint_self_world_prediction_v1.py

python test_joint_self_world_prediction_v1.py

set +e
python train_evaluate_joint_self_world_prediction_v1.py \
  --state-dir "$ROOT" \
  2>&1 | tee "$ROOT/run.log"
STATUS=${PIPESTATUS[0]}
set -e

if [ "$STATUS" -ne 0 ] && [ "$STATUS" -ne 4 ]; then
  echo "JOINT_SELF_WORLD_PREDICTION_WRAPPER: FAIL_EVALUATOR"
  exit "$STATUS"
fi

grep -q '^JOINT_SELF_WORLD_PREDICTION_RUN_COMPLETE$' "$ROOT/run.log"
grep -q '^PROTECTED_UNCHANGED: true$' "$ROOT/run.log"
grep -q '^V0293_TEST_STATUS_PRESERVED: FAIL$' "$ROOT/run.log"
grep -q '^JOINT_TEST_TOUCHED: false$' "$ROOT/run.log"

test -s "$ROOT/result.json"
test -s "$ROOT/joint_self_world_model.pt"

python - \
  "$ROOT/result.json" \
  "$ROOT/joint_self_world_model.pt" \
  "$ROOT/result_bundle.zip" <<'PY'
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import json, sys

r=Path(sys.argv[1])
m=Path(sys.argv[2])
b=Path(sys.argv[3])
p=json.loads(r.read_text())

assert p["protected_unchanged"] is True
assert p["v0293_test_status_preserved"] == "FAIL"
assert p["split"]["test_touched"] is False
assert p["training"]["joint_self_world_model"] is True
assert p["training"]["historical_brain"] is False
assert p["training"]["persistent_self"] is False
assert p["scope"]["subjective_experience_claimed"] is False
assert p["scope"]["sentience_claimed"] is False
assert p["scope"]["consciousness_claimed"] is False

with ZipFile(b,"w",ZIP_DEFLATED) as z:
    z.write(r,"result.json")
    z.write(m,"joint_self_world_model.pt")
    log=r.with_name("run.log")
    if log.is_file():
        z.write(log,"run.log")

print(b)
PY

if [ "$STATUS" -eq 0 ]; then
  echo "JOINT_SELF_WORLD_PREDICTION_DEV_GATE: PASS_CONFIRMED"
else
  echo "JOINT_SELF_WORLD_PREDICTION_DEV_GATE: FAIL_CONFIRMED"
fi

echo "JOINT_SELF_WORLD_PREDICTION_WRAPPER: COMPLETE"
exit "$STATUS"
