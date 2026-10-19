#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/continuous_brain_v0.12.2}"
cd "$PROJECT_ROOT"

ROOT="state/recurrent_goal_option_state_v1"
mkdir -p "$ROOT"
rm -f \
  "$ROOT/result.json" \
  "$ROOT/run.log" \
  "$ROOT/result_bundle.zip" \
  "$ROOT/recurrent_goal_option_state.pt"

echo "=== v0.22.2 Recurrent Goal-Option Temporal State ==="
echo "CPU reference backend."
echo "Trains only recurrent temporal option-state."
echo "Memory, goal decoder, goal-option policy, historical six and planner remain frozen."
echo "Uses full TRAIN 0-1599 and untouched TEST 1980-2359 exactly once."
echo "No TEST tuning is permitted after this run."

test -s state/memory_guided_goal_option_v1/goal_decoder.pt
test -s state/additive_goal_options_v1/goal_options.pt

python -m py_compile \
  continuous_brain/recurrent_goal_option_state_v1.py \
  train_test_recurrent_goal_option_state_v1.py

set +e
python train_test_recurrent_goal_option_state_v1.py \
  --state-dir "$ROOT" \
  2>&1 | tee "$ROOT/run.log"
STATUS=${PIPESTATUS[0]}
set -e

if [ "$STATUS" -ne 0 ] && [ "$STATUS" -ne 4 ]; then
  echo "RECURRENT_GOAL_OPTION_STATE_WRAPPER: FAIL_EVALUATOR"
  exit "$STATUS"
fi

grep -q '^RECURRENT_GOAL_OPTION_STATE_RUN_COMPLETE$' "$ROOT/run.log"
grep -q '^PROTECTED_UNCHANGED: true$' "$ROOT/run.log"
grep -q '^GOAL_OPTION_BANK_BIT_EXACT: true$' "$ROOT/run.log"
grep -q '^HISTORICAL_SIX_OPTIONS_BIT_EXACT: true$' "$ROOT/run.log"
grep -q '^V019_REMAINS_CLOSED: true$' "$ROOT/run.log"
grep -q '^DETOUR_PLANNER_REPAIR_CLOSED: true$' "$ROOT/run.log"
grep -q '^TEST_USED_FOR_TUNING_AFTER_RUN: false$' "$ROOT/run.log"

test -s "$ROOT/result.json"
test -s "$ROOT/recurrent_goal_option_state.pt"

python - "$ROOT/result.json" "$ROOT/recurrent_goal_option_state.pt" "$ROOT/result_bundle.zip" <<'PY'
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import json,sys

r=Path(sys.argv[1])
c=Path(sys.argv[2])
b=Path(sys.argv[3])
p=json.loads(r.read_text())

assert p["protected_unchanged"] is True
assert p["goal_option_bank_bit_exact"] is True
assert p["historical_six_options_bit_exact"] is True
assert p["split"]["test_used_once_for_admission"] is True
assert p["test_used_for_tuning_after_run"] is False

with ZipFile(b,"w",ZIP_DEFLATED) as z:
    z.write(r,"result.json")
    z.write(c,"recurrent_goal_option_state.pt")
    log=r.with_name("run.log")
    if log.is_file():
        z.write(log,"run.log")
print(b)
PY

if [ "$STATUS" -eq 0 ]; then
  echo "RECURRENT_GOAL_OPTION_STATE_TEST_ADMISSION: PASS_CONFIRMED"
else
  echo "RECURRENT_GOAL_OPTION_STATE_TEST_ADMISSION: FAIL_CONFIRMED"
fi

echo "RECURRENT_GOAL_OPTION_STATE_WRAPPER: COMPLETE"
exit "$STATUS"
