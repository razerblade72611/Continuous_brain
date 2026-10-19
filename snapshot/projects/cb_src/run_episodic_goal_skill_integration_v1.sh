#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/continuous_brain_v0.12.2}"
cd "$PROJECT_ROOT"

ROOT="state/episodic_goal_skill_integration_v1"
mkdir -p "$ROOT"
rm -f \
  "$ROOT/result.json" \
  "$ROOT/run.log" \
  "$ROOT/result_bundle.zip"

echo "=== v0.24.0 Episodic Selection -> Goal -> Temporal Skill ==="
echo "READ ONLY. CPU only. No training."
echo "Requires frozen v0.23 TEST PASS."
echo "Reuses historical TRAIN-family conditions only; consumes no new holdout."
echo "FIRST/SECOND select stored episodes; SAME/DIFFERENT remain relational operations."

test -s \
  state/multi_item_episodic_memory_test_v1/result.json
test -s \
  state/multi_item_episodic_memory_v1/episodic_reader.pt
test -s \
  state/memory_guided_goal_option_v1/goal_decoder.pt
test -s \
  state/additive_goal_options_v1/goal_options.pt
test -s \
  state/recurrent_goal_option_state_v1/recurrent_goal_option_state.pt

python -m py_compile \
  continuous_brain/episodic_goal_skill_integration_v1.py \
  evaluate_episodic_goal_skill_integration_v1.py

set +e
python evaluate_episodic_goal_skill_integration_v1.py \
  --state-dir "$ROOT" \
  2>&1 | tee "$ROOT/run.log"
STATUS=${PIPESTATUS[0]}
set -e

if [ "$STATUS" -ne 0 ] && [ "$STATUS" -ne 4 ]; then
  echo "EPISODIC_GOAL_SKILL_INTEGRATION_WRAPPER: FAIL_EVALUATOR"
  exit "$STATUS"
fi

grep -q '^EPISODIC_GOAL_SKILL_INTEGRATION_COMPLETE$' \
  "$ROOT/run.log"
grep -q '^PROTECTED_UNCHANGED: true$' \
  "$ROOT/run.log"
grep -q '^V023_READER_UNCHANGED: true$' \
  "$ROOT/run.log"
grep -q '^V0222_COMPONENTS_UNCHANGED: true$' \
  "$ROOT/run.log"
grep -q '^NEW_HOLDOUT_CONSUMED: false$' \
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
assert p["new_holdout_consumed"] is False
assert p["protected_unchanged"] is True
assert p["v023_reader_unchanged"] is True
assert p["v0222_components_unchanged"] is True
assert p["source"]["v023_composition_test_spent"] is True

with ZipFile(b, "w", ZIP_DEFLATED) as z:
    z.write(r, "result.json")
    log = r.with_name("run.log")
    if log.is_file():
        z.write(log, "run.log")

print(b)
PY

if [ "$STATUS" -eq 0 ]; then
  echo "EPISODIC_GOAL_SKILL_INTEGRATION_GATE: PASS_CONFIRMED"
else
  echo "EPISODIC_GOAL_SKILL_INTEGRATION_GATE: FAIL_CONFIRMED"
fi

echo "EPISODIC_GOAL_SKILL_INTEGRATION_WRAPPER: COMPLETE"
exit "$STATUS"
