#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/continuous_brain_v0.12.2}"
cd "$PROJECT_ROOT"

ROOT="state/multi_item_episodic_memory_test_v1"
mkdir -p "$ROOT"
rm -f \
  "$ROOT/result.json" \
  "$ROOT/run.log" \
  "$ROOT/result_bundle.zip"

echo "=== v0.23.1 Multi-Item Episodic Memory Frozen TEST ==="
echo "READ ONLY. CPU only. No training."
echo "Uses sealed composition TEST conditions 1312-1599 exactly once."
echo "After this run the composition TEST is spent and may not be tuned against."

test -s \
  state/multi_item_episodic_memory_v1/result.json
test -s \
  state/multi_item_episodic_memory_v1/episodic_reader.pt
test -s \
  state/milestones/v0222_memory_goal_temporal_skill/STATUS.txt

python -m py_compile \
  evaluate_multi_item_episodic_memory_test_v1.py

set +e
python evaluate_multi_item_episodic_memory_test_v1.py \
  --state-dir "$ROOT" \
  2>&1 | tee "$ROOT/run.log"
STATUS=${PIPESTATUS[0]}
set -e

if [ "$STATUS" -ne 0 ] && [ "$STATUS" -ne 4 ]; then
  echo "MULTI_ITEM_EPISODIC_MEMORY_TEST_WRAPPER: FAIL_EVALUATOR"
  exit "$STATUS"
fi

grep -q '^MULTI_ITEM_EPISODIC_MEMORY_TEST_COMPLETE$' \
  "$ROOT/run.log"
grep -q '^PROTECTED_UNCHANGED: true$' \
  "$ROOT/run.log"
grep -q '^V0222_MILESTONE_UNTOUCHED: true$' \
  "$ROOT/run.log"
grep -q '^COMPOSITION_TEST_SPENT: true$' \
  "$ROOT/run.log"
grep -q '^TEST_TUNING_AFTER_RUN: forbidden$' \
  "$ROOT/run.log"
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
assert p["v0222_milestone_untouched"] is True
assert p["composition_test_spent"] is True
assert p["split"]["test_tuning_after_run_forbidden"] is True

with ZipFile(b, "w", ZIP_DEFLATED) as z:
    z.write(r, "result.json")
    log = r.with_name("run.log")
    if log.is_file():
        z.write(log, "run.log")

print(b)
PY

if [ "$STATUS" -eq 0 ]; then
  echo "MULTI_ITEM_EPISODIC_MEMORY_TEST_ADMISSION: PASS_CONFIRMED"
else
  echo "MULTI_ITEM_EPISODIC_MEMORY_TEST_ADMISSION: FAIL_CONFIRMED"
fi

echo "MULTI_ITEM_EPISODIC_MEMORY_TEST_WRAPPER: COMPLETE"
exit "$STATUS"
