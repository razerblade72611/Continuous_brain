#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/continuous_brain_v0.12.2}"
cd "$PROJECT_ROOT"

ROOT="state/goal_option_progress_v1"
mkdir -p "$ROOT"
rm -f "$ROOT/result.json" "$ROOT/run.log" "$ROOT/result_bundle.zip" "$ROOT/progress_head.pt"

echo "=== v0.22.1 Goal-Option Temporal Progress Repair ==="
echo "CPU reference backend."
echo "Freezes memory, goal decoder, new goal-option policy, historical six, planner."
echo "Trains only a dense remaining-duration head."
echo "Uses fresh DEV conditions 1856-1979. TEST remains sealed."

test -s state/memory_guided_goal_option_v1/goal_decoder.pt
test -s state/additive_goal_options_v1/goal_options.pt

python -m py_compile \
  continuous_brain/goal_option_progress_v1.py \
  train_evaluate_goal_option_progress_v1.py

set +e
python train_evaluate_goal_option_progress_v1.py \
  --state-dir "$ROOT" \
  2>&1 | tee "$ROOT/run.log"
STATUS=${PIPESTATUS[0]}
set -e

if [ "$STATUS" -ne 0 ] && [ "$STATUS" -ne 4 ]; then
  echo "GOAL_OPTION_TEMPORAL_PROGRESS_WRAPPER: FAIL_EVALUATOR"
  exit "$STATUS"
fi

grep -q '^GOAL_OPTION_TEMPORAL_PROGRESS_RUN_COMPLETE$' "$ROOT/run.log"
grep -q '^PROTECTED_UNCHANGED: true$' "$ROOT/run.log"
grep -q '^GOAL_OPTION_BANK_BIT_EXACT: true$' "$ROOT/run.log"
grep -q '^HISTORICAL_SIX_OPTIONS_BIT_EXACT: true$' "$ROOT/run.log"
grep -q '^TEST_TOUCHED: false$' "$ROOT/run.log"
grep -q '^V019_REMAINS_CLOSED: true$' "$ROOT/run.log"
grep -q '^DETOUR_PLANNER_REPAIR_CLOSED: true$' "$ROOT/run.log"
test -s "$ROOT/result.json"
test -s "$ROOT/progress_head.pt"

python - "$ROOT/result.json" "$ROOT/progress_head.pt" "$ROOT/result_bundle.zip" <<'PY'
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import json, sys

r=Path(sys.argv[1]); h=Path(sys.argv[2]); b=Path(sys.argv[3])
p=json.loads(r.read_text())

assert p["protected_unchanged"] is True
assert p["goal_option_bank_bit_exact"] is True
assert p["historical_six_options_bit_exact"] is True
assert p["split"]["test_touched"] is False
assert p["training"]["temporal_progress_head"] is True
assert p["training"]["goal_option_policy"] is False
assert p["training"]["goal_option_bank"] is False
assert p["training"]["historical_options"] is False
assert p["training"]["memory"] is False
assert p["training"]["planner"] is False

with ZipFile(b,"w",ZIP_DEFLATED) as z:
    z.write(r,"result.json")
    z.write(h,"progress_head.pt")
    log=r.with_name("run.log")
    if log.is_file():
        z.write(log,"run.log")
print(b)
PY

if [ "$STATUS" -eq 0 ]; then
  echo "GOAL_OPTION_TEMPORAL_PROGRESS_DEV_GATE: PASS_CONFIRMED"
else
  echo "GOAL_OPTION_TEMPORAL_PROGRESS_DEV_GATE: FAIL_CONFIRMED"
fi

echo "GOAL_OPTION_TEMPORAL_PROGRESS_WRAPPER: COMPLETE"
exit "$STATUS"
