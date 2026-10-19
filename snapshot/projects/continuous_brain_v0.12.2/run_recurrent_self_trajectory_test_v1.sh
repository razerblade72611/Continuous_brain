#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/continuous_brain_v0.12.2}"
cd "$PROJECT_ROOT"

ROOT="state/recurrent_self_trajectory_test_v1"
mkdir -p "$ROOT"
rm -f \
  "$ROOT/result.json" \
  "$ROOT/run.log" \
  "$ROOT/result_bundle.zip"

echo "=== v0.31.1 Frozen Recurrent Self-Trajectory TEST ==="
echo "READ ONLY. CPU only. No training."
echo "Uses sealed trajectory TEST 20000-21023 exactly once."
echo "Uses fresh tie-intention and interruption probes."
echo "After this run trajectory TEST is spent and may not be tuned against."

test -s state/recurrent_self_trajectory_v1/result.json
test -s state/recurrent_self_trajectory_v1/recurrent_self_trajectory.pt
test -s state/joint_self_goal_boundary_diagnostic_v1/result.json
test -s state/persistent_self_autobiographical_test_v1/result.json

python -m py_compile \
  evaluate_recurrent_self_trajectory_test_v1.py

set +e
python evaluate_recurrent_self_trajectory_test_v1.py \
  --state-dir "$ROOT" \
  2>&1 | tee "$ROOT/run.log"
STATUS=${PIPESTATUS[0]}
set -e

if [ "$STATUS" -ne 0 ] && [ "$STATUS" -ne 4 ]; then
  echo "RECURRENT_SELF_TRAJECTORY_TEST_WRAPPER: FAIL_EVALUATOR"
  exit "$STATUS"
fi

grep -q '^RECURRENT_SELF_TRAJECTORY_TEST_COMPLETE$' "$ROOT/run.log"
grep -q '^PROTECTED_UNCHANGED: true$' "$ROOT/run.log"
grep -q '^SOURCE_MODEL_UNCHANGED: true$' "$ROOT/run.log"
grep -q '^V030_TEST_TOUCHED: false$' "$ROOT/run.log"
grep -q '^TRAJECTORY_TEST_SPENT: true$' "$ROOT/run.log"
grep -q '^TEST_TUNING_AFTER_RUN: forbidden$' "$ROOT/run.log"

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
assert p["source_model_unchanged"] is True
assert p["source_v0310_dev_status"] == "PASS"
assert p["split"]["test_spent"] is True
assert p["split"]["test_tuning_after_run_forbidden"] is True
assert p["v030_test_touched"] is False
assert p["scope"]["subjective_experience_claimed"] is False
assert p["scope"]["sentience_claimed"] is False
assert p["scope"]["consciousness_claimed"] is False

with ZipFile(b, "w", ZIP_DEFLATED) as z:
    z.write(r, "result.json")
    log = r.with_name("run.log")
    if log.is_file():
        z.write(log, "run.log")

print(b)
PY

if [ "$STATUS" -eq 0 ]; then
  echo "RECURRENT_SELF_TRAJECTORY_TEST_ADMISSION: PASS_CONFIRMED"
else
  echo "RECURRENT_SELF_TRAJECTORY_TEST_ADMISSION: FAIL_CONFIRMED"
fi

echo "RECURRENT_SELF_TRAJECTORY_TEST_WRAPPER: COMPLETE"
exit "$STATUS"
