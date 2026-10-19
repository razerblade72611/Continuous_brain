#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/continuous_brain_v0.12.2}"
cd "$PROJECT_ROOT"

ROOT="state/endogenous_goal_genesis_v1"
mkdir -p "$ROOT"
rm -f \
  "$ROOT/result.json" \
  "$ROOT/run.log" \
  "$ROOT/result_bundle.zip" \
  "$ROOT/goal_lifecycle_probe.pt"

echo "=== v0.35.0 Endogenous Autobiographical Goal Genesis ==="
echo "Cognition milestone. No robot training. No neural training."
echo "Requires frozen v0.34.1 TEST PASS."
echo "No external A/B goal is supplied to the goal generator."
echo "The system authors a developmental target from its own autobiographical self-estimate."
echo "Fresh DEV subjects 90000-90127."
echo "Fresh TEST subjects 100000-100127 remain sealed."

test -s state/event_consistent_self_development_test_v1/result.json
test -s state/autobiographical_self_discrepancy_neutral_v1/neutral_autobiographical_self_model.pt
test -s state/recurrent_self_trajectory_v1/recurrent_self_trajectory.pt

python -m py_compile \
  continuous_brain/endogenous_goal_genesis_v1.py \
  evaluate_endogenous_goal_genesis_v1.py \
  test_endogenous_goal_genesis_v1.py

python test_endogenous_goal_genesis_v1.py

set +e
python evaluate_endogenous_goal_genesis_v1.py \
  --state-dir "$ROOT" \
  2>&1 | tee "$ROOT/run.log"
STATUS=${PIPESTATUS[0]}
set -e

if [ "$STATUS" -ne 0 ] && [ "$STATUS" -ne 4 ]; then
  echo "ENDOGENOUS_GOAL_GENESIS_WRAPPER: FAIL_EVALUATOR"
  exit "$STATUS"
fi

grep -q '^ENDOGENOUS_GOAL_GENESIS_RUN_COMPLETE$' "$ROOT/run.log"
grep -q '^TRAINING_PERFORMED: false$' "$ROOT/run.log"
grep -q '^PROTECTED_UNCHANGED: true$' "$ROOT/run.log"
grep -q '^V0341_TEST_PASS_PRESERVED: true$' "$ROOT/run.log"
grep -q '^V035_TEST_TOUCHED: false$' "$ROOT/run.log"

test -s "$ROOT/result.json"

python - "$ROOT/result.json" "$ROOT/result_bundle.zip" <<'PY'
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import json, sys

r = Path(sys.argv[1])
b = Path(sys.argv[2])
p = json.loads(r.read_text())

assert p["family"] == "v0.35"
assert p["training_performed"] is False
assert p["source_v0341_test_status"] == "PASS"
assert p["architecture"]["external_primary_goal_supplied_to_generator"] is False
assert p["architecture"]["hidden_capability_supplied_to_generator"] is False
assert p["architecture"]["subject_id_supplied_to_generator"] is False
assert p["architecture"]["reward_label_supplied_to_generator"] is False
assert p["split"]["test_touched"] is False
assert p["protected_unchanged"] is True
assert p["scope"]["open_ended_goal_space_claimed"] is False
assert p["scope"]["subjective_experience_claimed"] is False
assert p["scope"]["sentience_claimed"] is False
assert p["scope"]["consciousness_claimed"] is False

with ZipFile(b, "w", ZIP_DEFLATED) as z:
    z.write(r, "result.json")
    log = r.with_name("run.log")
    if log.is_file():
        z.write(log, "run.log")
    goal = r.with_name("goal_lifecycle_probe.pt")
    if goal.is_file():
        z.write(goal, "goal_lifecycle_probe.pt")

print(b)
PY

if [ "$STATUS" -eq 0 ]; then
  echo "ENDOGENOUS_GOAL_GENESIS_DEV_GATE: PASS_CONFIRMED"
else
  echo "ENDOGENOUS_GOAL_GENESIS_DEV_GATE: FAIL_CONFIRMED"
fi

echo "ENDOGENOUS_GOAL_GENESIS_WRAPPER: COMPLETE"
exit "$STATUS"
