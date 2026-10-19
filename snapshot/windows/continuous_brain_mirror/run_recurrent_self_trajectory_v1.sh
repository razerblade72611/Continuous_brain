#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/continuous_brain_v0.12.2}"
cd "$PROJECT_ROOT"

ROOT="state/recurrent_self_trajectory_v1"
mkdir -p "$ROOT"
rm -f \
  "$ROOT/result.json" \
  "$ROOT/run.log" \
  "$ROOT/result_bundle.zip" \
  "$ROOT/recurrent_self_trajectory.pt"

echo "=== v0.31.0 Recurrent Imagined Self-Trajectory ==="
echo "Cognition milestone. No robot training."
echo "New development family after v0.30 boundary diagnostic."
echo "Carries primary intention separately from temporary INVESTIGATE mode."
echo "Predicts multi-step future world + self trajectories."
echo "Trajectory TEST 20000-21023 remains sealed."

test -s state/joint_self_world_prediction_v1/result.json
test -s state/joint_self_world_repair_v1/result.json
test -s state/joint_self_goal_boundary_diagnostic_v1/result.json
test -s state/persistent_self_autobiographical_test_v1/result.json

python -m py_compile \
  continuous_brain/recurrent_self_trajectory_v1.py \
  train_evaluate_recurrent_self_trajectory_v1.py \
  test_recurrent_self_trajectory_v1.py

python test_recurrent_self_trajectory_v1.py

set +e
python train_evaluate_recurrent_self_trajectory_v1.py \
  --state-dir "$ROOT" \
  2>&1 | tee "$ROOT/run.log"
STATUS=${PIPESTATUS[0]}
set -e

if [ "$STATUS" -ne 0 ] && [ "$STATUS" -ne 4 ]; then
  echo "RECURRENT_SELF_TRAJECTORY_WRAPPER: FAIL_EVALUATOR"
  exit "$STATUS"
fi

grep -q '^RECURRENT_SELF_TRAJECTORY_RUN_COMPLETE$' "$ROOT/run.log"
grep -q '^PROTECTED_UNCHANGED: true$' "$ROOT/run.log"
grep -q '^V030_TEST_TOUCHED: false$' "$ROOT/run.log"
grep -q '^TRAJECTORY_TEST_TOUCHED: false$' "$ROOT/run.log"

test -s "$ROOT/result.json"
test -s "$ROOT/recurrent_self_trajectory.pt"

python - \
  "$ROOT/result.json" \
  "$ROOT/recurrent_self_trajectory.pt" \
  "$ROOT/result_bundle.zip" <<'PY'
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import json,sys

r=Path(sys.argv[1])
m=Path(sys.argv[2])
b=Path(sys.argv[3])
p=json.loads(r.read_text())

assert p["protected_unchanged"] is True
assert p["trajectory_test_touched"] is False
assert p["historical_status"]["v030_test_touched"] is False
assert p["training"]["recurrent_self_trajectory_model"] is True
assert p["training"]["historical_brain"] is False
assert p["training"]["persistent_self"] is False
assert p["architecture"]["explicit_primary_intention"] is True
assert p["architecture"]["temporary_investigate_mode"] is True
assert p["scope"]["subjective_experience_claimed"] is False
assert p["scope"]["sentience_claimed"] is False
assert p["scope"]["consciousness_claimed"] is False

with ZipFile(b,"w",ZIP_DEFLATED) as z:
    z.write(r,"result.json")
    z.write(m,"recurrent_self_trajectory.pt")
    log=r.with_name("run.log")
    if log.is_file():
        z.write(log,"run.log")
print(b)
PY

if [ "$STATUS" -eq 0 ]; then
  echo "RECURRENT_SELF_TRAJECTORY_DEV_GATE: PASS_CONFIRMED"
else
  echo "RECURRENT_SELF_TRAJECTORY_DEV_GATE: FAIL_CONFIRMED"
fi

echo "RECURRENT_SELF_TRAJECTORY_WRAPPER: COMPLETE"
exit "$STATUS"
