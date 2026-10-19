#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/continuous_brain_v0.12.2}"
cd "$PROJECT_ROOT"

ROOT="state/memory_guided_goal_option_v1"
mkdir -p "$ROOT"
rm -f "$ROOT/result.json" "$ROOT/run.log" "$ROOT/result_bundle.zip"

echo "=== v0.21.0 Memory-Guided Goal Option Control ==="
echo "CPU reference backend."
echo "Leaves Detour planner repair closed."
echo "Trains only a 128->2 remembered-goal decoder."
echo "Base, admitted memory, options, planner and v0.19 remain frozen."

python -m py_compile \
  continuous_brain/memory_guided_goal_option_v1.py \
  run_memory_guided_goal_option_v1.py

set +e
python run_memory_guided_goal_option_v1.py \
  --state-dir "$ROOT" \
  2>&1 | tee "$ROOT/run.log"
STATUS=${PIPESTATUS[0]}
set -e

# Exit 4 is a valid scientific gate failure, not a wrapper/software failure.
if [ "$STATUS" -ne 0 ] && [ "$STATUS" -ne 4 ]; then
  echo "MEMORY_GUIDED_GOAL_OPTION_WRAPPER: FAIL_EVALUATOR"
  exit "$STATUS"
fi

grep -q '^MEMORY_GUIDED_GOAL_OPTION_RUN_COMPLETE$' "$ROOT/run.log"
grep -q '^PROTECTED_UNCHANGED: true$' "$ROOT/run.log"
grep -q '^TEST_TOUCHED: false$' "$ROOT/run.log"
grep -q '^V019_REMAINS_CLOSED: true$' "$ROOT/run.log"
grep -q '^DETOUR_PLANNER_REPAIR_CLOSED: true$' "$ROOT/run.log"
test -s "$ROOT/result.json"
test -s "$ROOT/goal_decoder.pt"

python - "$ROOT/result.json" "$ROOT/goal_decoder.pt" "$ROOT/result_bundle.zip" <<'PY'
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import json, sys

result = Path(sys.argv[1])
decoder = Path(sys.argv[2])
bundle = Path(sys.argv[3])

p = json.loads(result.read_text())
assert p["protected_unchanged"] is True
assert p["split"]["test_touched"] is False
assert p["v019_remains_closed"] is True
assert p["detour_planner_repair_closed"] is True
assert p["training"]["goal_decoder_only"] is True
assert p["training"]["base"] is False
assert p["training"]["memory"] is False
assert p["training"]["options"] is False
assert p["training"]["planner"] is False

with ZipFile(bundle, "w", ZIP_DEFLATED) as z:
    z.write(result, "result.json")
    z.write(decoder, "goal_decoder.pt")
    log = result.with_name("run.log")
    if log.is_file():
        z.write(log, "run.log")

print(bundle)
PY

if [ "$STATUS" -eq 0 ]; then
  echo "MEMORY_GUIDED_GOAL_OPTION_DEV_GATE: PASS_CONFIRMED"
else
  echo "MEMORY_GUIDED_GOAL_OPTION_DEV_GATE: FAIL_CONFIRMED"
fi

echo "MEMORY_GUIDED_GOAL_OPTION_WRAPPER: COMPLETE"
exit "$STATUS"
