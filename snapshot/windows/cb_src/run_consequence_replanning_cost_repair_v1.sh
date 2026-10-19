#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/continuous_brain_v0.12.2}"
cd "$PROJECT_ROOT"

ROOT="state/consequence_replanning_cost_repair_v1"
mkdir -p "$ROOT"
rm -f "$ROOT/result.json" "$ROOT/run.log" "$ROOT/result_bundle.zip"

echo "=== v0.25.1 Consequence-Replanning Cost-Consistent Planner Repair ==="
echo "READ ONLY. CPU only. No training."
echo "Reuses v0.25.0 consequence model and spent DEV 2048-2303."
echo "Fixes only planner ranking: utility first, then success probability."
echo "Adds explicit surprise-exposure invariant."
echo "Route TEST 2304-2559 remains sealed."

test -s state/consequence_replanning_v1/result.json
test -s state/consequence_replanning_v1/consequence_model.pt
test -s state/multi_item_episodic_memory_v1/episodic_reader.pt
test -s state/memory_guided_goal_option_v1/goal_decoder.pt

python -m py_compile \
  continuous_brain/consequence_replanning_cost_repair_v1.py \
  evaluate_consequence_replanning_cost_repair_v1.py

set +e
python evaluate_consequence_replanning_cost_repair_v1.py \
  --state-dir "$ROOT" \
  2>&1 | tee "$ROOT/run.log"
STATUS=${PIPESTATUS[0]}
set -e

if [ "$STATUS" -ne 0 ] && [ "$STATUS" -ne 4 ]; then
  echo "CONSEQUENCE_REPLANNING_COST_REPAIR_WRAPPER: FAIL_EVALUATOR"
  exit "$STATUS"
fi

grep -q '^CONSEQUENCE_REPLANNING_COST_REPAIR_COMPLETE$' "$ROOT/run.log"
grep -q '^PROTECTED_UNCHANGED: true$' "$ROOT/run.log"
grep -q '^SOURCE_CONSEQUENCE_MODEL_UNCHANGED: true$' "$ROOT/run.log"
grep -q '^HISTORICAL_PLANNER_UNTOUCHED: true$' "$ROOT/run.log"
grep -q '^ROUTE_TEST_TOUCHED: false$' "$ROOT/run.log"
test -s "$ROOT/result.json"

python - "$ROOT/result.json" "$ROOT/result_bundle.zip" <<'PY'
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import json, sys

r=Path(sys.argv[1]); b=Path(sys.argv[2])
p=json.loads(r.read_text())
assert p["training_performed"] is False
assert p["protected_unchanged"] is True
assert p["source_consequence_model_unchanged"] is True
assert p["historical_planner_untouched"] is True
assert p["split"]["route_test_touched"] is False
assert p["split"]["new_development_window_consumed"] is False

with ZipFile(b,"w",ZIP_DEFLATED) as z:
    z.write(r,"result.json")
    log=r.with_name("run.log")
    if log.is_file():
        z.write(log,"run.log")
print(b)
PY

if [ "$STATUS" -eq 0 ]; then
  echo "CONSEQUENCE_REPLANNING_COST_REPAIR_DEV_GATE: PASS_CONFIRMED"
else
  echo "CONSEQUENCE_REPLANNING_COST_REPAIR_DEV_GATE: FAIL_CONFIRMED"
fi

echo "CONSEQUENCE_REPLANNING_COST_REPAIR_WRAPPER: COMPLETE"
exit "$STATUS"
