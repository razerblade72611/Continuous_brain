#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/continuous_brain_v0.12.2}"
cd "$PROJECT_ROOT"

ROOT="state/multi_item_episodic_memory_v1"
mkdir -p "$ROOT"
rm -f \
  "$ROOT/result.json" \
  "$ROOT/run.log" \
  "$ROOT/result_bundle.zip" \
  "$ROOT/episodic_reader.pt"

echo "=== v0.23.0 Multi-Item Episodic Working Memory ==="
echo "CPU reference backend."
echo "Preserves admitted 64-D writer + canonical recall."
echo "Adds two ordered canonical-recall slots and a late-query reader."
echo "Queries: FIRST / SECOND / SAME / DIFFERENT."
echo "Composition TEST remains sealed."

test -s \
  state/working_memory_recall_capacity_control_phase1_v1/latest.portable.pt
test -s \
  state/memory_guided_goal_option_v1/goal_decoder.pt
test -s \
  state/milestones/v0222_memory_goal_temporal_skill/STATUS.txt

python -m py_compile \
  continuous_brain/multi_item_episodic_memory_v1.py \
  train_evaluate_multi_item_episodic_memory_v1.py \
  test_multi_item_episodic_memory_v1.py

python test_multi_item_episodic_memory_v1.py

set +e
python train_evaluate_multi_item_episodic_memory_v1.py \
  --state-dir "$ROOT" \
  2>&1 | tee "$ROOT/run.log"
STATUS=${PIPESTATUS[0]}
set -e

if [ "$STATUS" -ne 0 ] && [ "$STATUS" -ne 4 ]; then
  echo "MULTI_ITEM_EPISODIC_MEMORY_WRAPPER: FAIL_EVALUATOR"
  exit "$STATUS"
fi

grep -q '^MULTI_ITEM_EPISODIC_MEMORY_RUN_COMPLETE$' \
  "$ROOT/run.log"
grep -q '^PROTECTED_UNCHANGED: true$' \
  "$ROOT/run.log"
grep -q '^V0222_MILESTONE_UNTOUCHED: true$' \
  "$ROOT/run.log"
grep -q '^COMPOSITION_TEST_TOUCHED: false$' \
  "$ROOT/run.log"

test -s "$ROOT/result.json"
test -s "$ROOT/episodic_reader.pt"

python - "$ROOT/result.json" \
  "$ROOT/episodic_reader.pt" \
  "$ROOT/result_bundle.zip" <<'PY'
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import json, sys

r = Path(sys.argv[1])
c = Path(sys.argv[2])
b = Path(sys.argv[3])

p = json.loads(r.read_text())
assert p["protected_unchanged"] is True
assert p["v0222_milestone_untouched"] is True
assert p["split"]["composition_test_touched"] is False
assert p["training"]["late_query_reader"] is True
assert p["training"]["historical_64d_writer"] is False
assert p["training"]["canonical_recall"] is False
assert p["training"]["base"] is False

with ZipFile(b, "w", ZIP_DEFLATED) as z:
    z.write(r, "result.json")
    z.write(c, "episodic_reader.pt")
    log = r.with_name("run.log")
    if log.is_file():
        z.write(log, "run.log")

print(b)
PY

if [ "$STATUS" -eq 0 ]; then
  echo "MULTI_ITEM_EPISODIC_MEMORY_DEV_GATE: PASS_CONFIRMED"
else
  echo "MULTI_ITEM_EPISODIC_MEMORY_DEV_GATE: FAIL_CONFIRMED"
fi

echo "MULTI_ITEM_EPISODIC_MEMORY_WRAPPER: COMPLETE"
exit "$STATUS"
