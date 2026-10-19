#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/continuous_brain_v0.12.2}"
cd "$PROJECT_ROOT"

ROOT="state/consequence_replanning_v1"
mkdir -p "$ROOT"
rm -f \
  "$ROOT/result.json" \
  "$ROOT/run.log" \
  "$ROOT/result_bundle.zip" \
  "$ROOT/consequence_model.pt"

echo "=== v0.25.0 Episodic-Goal Consequence Planning + Replanning ==="
echo "CPU reference backend."
echo "New task family: DynamicRouteGraph v1."
echo "Trains only a new short-horizon macro consequence model."
echo "Historical planner/world model remain untouched."
echo "Route TEST 2304-2559 remains sealed."

test -s state/multi_item_episodic_memory_test_v1/result.json
test -s state/multi_item_episodic_memory_v1/episodic_reader.pt
test -s state/memory_guided_goal_option_v1/goal_decoder.pt
test -s state/episodic_goal_skill_integration_v1/result.json

python -m py_compile \
  continuous_brain/dynamic_route_world_v1.py \
  continuous_brain/consequence_replanning_v1.py \
  train_evaluate_consequence_replanning_v1.py \
  test_consequence_replanning_v1.py

python test_consequence_replanning_v1.py

set +e
python train_evaluate_consequence_replanning_v1.py \
  --state-dir "$ROOT" \
  2>&1 | tee "$ROOT/run.log"
STATUS=${PIPESTATUS[0]}
set -e

if [ "$STATUS" -ne 0 ] && [ "$STATUS" -ne 4 ]; then
  echo "CONSEQUENCE_REPLANNING_WRAPPER: FAIL_EVALUATOR"
  exit "$STATUS"
fi

grep -q '^CONSEQUENCE_REPLANNING_RUN_COMPLETE$' "$ROOT/run.log"
grep -q '^PROTECTED_UNCHANGED: true$' "$ROOT/run.log"
grep -q '^V023_MILESTONE_UNCHANGED: true$' "$ROOT/run.log"
grep -q '^V024_INTEGRATION_UNCHANGED: true$' "$ROOT/run.log"
grep -q '^ROUTE_TEST_TOUCHED: false$' "$ROOT/run.log"

test -s "$ROOT/result.json"
test -s "$ROOT/consequence_model.pt"

python - "$ROOT/result.json" "$ROOT/consequence_model.pt" "$ROOT/result_bundle.zip" <<'PY'
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import json,sys

r=Path(sys.argv[1]); m=Path(sys.argv[2]); b=Path(sys.argv[3])
p=json.loads(r.read_text())

assert p["protected_unchanged"] is True
assert p["v023_milestone_unchanged"] is True
assert p["v024_integration_unchanged"] is True
assert p["split"]["route_test_touched"] is False
assert p["training"]["route_consequence_model"] is True
assert p["training"]["historical_planner"] is False
assert p["training"]["historical_world_model"] is False
assert p["training"]["memory"] is False
assert p["training"]["episodic_reader"] is False
assert p["training"]["goal_decoder"] is False

with ZipFile(b,"w",ZIP_DEFLATED) as z:
    z.write(r,"result.json")
    z.write(m,"consequence_model.pt")
    log=r.with_name("run.log")
    if log.is_file(): z.write(log,"run.log")
print(b)
PY

if [ "$STATUS" -eq 0 ]; then
  echo "CONSEQUENCE_REPLANNING_DEV_GATE: PASS_CONFIRMED"
else
  echo "CONSEQUENCE_REPLANNING_DEV_GATE: FAIL_CONFIRMED"
fi

echo "CONSEQUENCE_REPLANNING_WRAPPER: COMPLETE"
exit "$STATUS"
