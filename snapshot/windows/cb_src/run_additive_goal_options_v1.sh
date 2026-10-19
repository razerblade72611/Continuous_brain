#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/continuous_brain_v0.12.2}"
cd "$PROJECT_ROOT"

ROOT="state/additive_goal_options_v1"
mkdir -p "$ROOT"
rm -f "$ROOT/result.json" "$ROOT/run.log" "$ROOT/result_bundle.zip" "$ROOT/goal_options.pt"

echo "=== v0.22.0 Additive Goal-Diverse Temporal Options ==="
echo "CPU reference backend."
echo "Adds two sidecar temporal options; historical six remain frozen."
echo "Trains policy + termination for the new option bank only."
echo "Uses fresh DEV conditions 1728-1855. TEST remains sealed."

test -s state/memory_guided_goal_option_v1/goal_decoder.pt

python -m py_compile \
  continuous_brain/additive_goal_options_v1.py \
  train_evaluate_additive_goal_options_v1.py

set +e
python train_evaluate_additive_goal_options_v1.py \
  --state-dir "$ROOT" \
  2>&1 | tee "$ROOT/run.log"
STATUS=${PIPESTATUS[0]}
set -e

if [ "$STATUS" -ne 0 ] && [ "$STATUS" -ne 4 ]; then
  echo "ADDITIVE_GOAL_OPTIONS_WRAPPER: FAIL_EVALUATOR"
  exit "$STATUS"
fi

grep -q '^ADDITIVE_GOAL_OPTIONS_RUN_COMPLETE$' "$ROOT/run.log"
grep -q '^PROTECTED_UNCHANGED: true$' "$ROOT/run.log"
grep -q '^HISTORICAL_SIX_OPTIONS_BIT_EXACT: true$' "$ROOT/run.log"
grep -q '^TEST_TOUCHED: false$' "$ROOT/run.log"
grep -q '^V019_REMAINS_CLOSED: true$' "$ROOT/run.log"
grep -q '^DETOUR_PLANNER_REPAIR_CLOSED: true$' "$ROOT/run.log"
test -s "$ROOT/result.json"
test -s "$ROOT/goal_options.pt"

python - "$ROOT/result.json" "$ROOT/goal_options.pt" "$ROOT/result_bundle.zip" <<'PY'
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import json,sys
r=Path(sys.argv[1]); c=Path(sys.argv[2]); b=Path(sys.argv[3])
p=json.loads(r.read_text())
assert p["protected_unchanged"] is True
assert p["historical_six_options_bit_exact"] is True
assert p["split"]["test_touched"] is False
assert p["training"]["historical_options"] is False
assert p["training"]["memory"] is False
assert p["training"]["base"] is False
assert p["training"]["planner"] is False
with ZipFile(b,"w",ZIP_DEFLATED) as z:
    z.write(r,"result.json")
    z.write(c,"goal_options.pt")
    log=r.with_name("run.log")
    if log.is_file(): z.write(log,"run.log")
print(b)
PY

if [ "$STATUS" -eq 0 ]; then
  echo "ADDITIVE_GOAL_OPTIONS_DEV_GATE: PASS_CONFIRMED"
else
  echo "ADDITIVE_GOAL_OPTIONS_DEV_GATE: FAIL_CONFIRMED"
fi
echo "ADDITIVE_GOAL_OPTIONS_WRAPPER: COMPLETE"
exit "$STATUS"
