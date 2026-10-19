#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/continuous_brain_v0.12.2}"
cd "$PROJECT_ROOT"

ROOT="state/endogenous_goal_genesis_test_v1"
mkdir -p "$ROOT"
rm -f \
  "$ROOT/result.json" \
  "$ROOT/run.log" \
  "$ROOT/result_bundle.zip" \
  "$ROOT/goal_lifecycle_probe.pt"

echo "=== v0.35.3 Frozen Endogenous Goal Genesis TEST ==="
echo "READ ONLY. CPU only. No training."
echo "Uses exact v0.35.2 goal generator, +0.12 completion criterion, and 16-episode horizon."
echo "Uses sealed TEST subjects 100000-100127 exactly once."
echo "After this run the v0.35 TEST is permanently spent."
echo "No tuning after TEST."

test -s state/endogenous_goal_horizon_repair_v1/result.json
test -s state/endogenous_goal_genesis_v1/result.json
test -s state/endogenous_goal_completion_diagnostic_v1/result.json
test -s state/event_consistent_self_development_test_v1/result.json
test -s state/autobiographical_self_discrepancy_neutral_v1/neutral_autobiographical_self_model.pt
test -s state/recurrent_self_trajectory_v1/recurrent_self_trajectory.pt

python -m py_compile \
  evaluate_endogenous_goal_genesis_test_v1.py \
  test_endogenous_goal_genesis_test_v1.py

python test_endogenous_goal_genesis_test_v1.py

set +e
python evaluate_endogenous_goal_genesis_test_v1.py \
  --state-dir "$ROOT" \
  2>&1 | tee "$ROOT/run.log"
STATUS=${PIPESTATUS[0]}
set -e

if [ "$STATUS" -ne 0 ] && [ "$STATUS" -ne 4 ]; then
  echo "ENDOGENOUS_GOAL_GENESIS_TEST_WRAPPER: FAIL_EVALUATOR"
  exit "$STATUS"
fi

grep -q '^ENDOGENOUS_GOAL_GENESIS_TEST_COMPLETE$' "$ROOT/run.log"
grep -q '^TRAINING_PERFORMED: false$' "$ROOT/run.log"
grep -q '^GOAL_GENERATOR_CHANGED: false$' "$ROOT/run.log"
grep -q '^GOAL_COMPLETION_THRESHOLD_CHANGED: false$' "$ROOT/run.log"
grep -q '^PROTECTED_UNCHANGED: true$' "$ROOT/run.log"
grep -q '^SOURCE_MODELS_UNCHANGED: true$' "$ROOT/run.log"
grep -q '^V035_TEST_SPENT: true$' "$ROOT/run.log"
grep -q '^TEST_TUNING_AFTER_RUN: forbidden$' "$ROOT/run.log"

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
assert p["historical_status"]["v0350"] == "DEV_FAIL"
assert p["historical_status"]["v0352"] == "DEV_PASS"
assert p["frozen_mechanism"]["goal_generator_changed"] is False
assert p["frozen_mechanism"]["goal_completion_threshold_changed"] is False
assert p["frozen_mechanism"]["goal_target_delta"] == 0.12
assert p["frozen_mechanism"]["max_target_episodes"] == 16
assert p["frozen_mechanism"]["external_primary_goal_supplied_to_generator"] is False
assert p["split"]["test_spent"] is True
assert p["split"]["test_tuning_after_run_forbidden"] is True
assert p["protected_unchanged"] is True
assert p["source_models_unchanged"] is True
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
  echo "ENDOGENOUS_GOAL_GENESIS_TEST_ADMISSION: PASS_CONFIRMED"
else
  echo "ENDOGENOUS_GOAL_GENESIS_TEST_ADMISSION: FAIL_CONFIRMED"
fi

echo "ENDOGENOUS_GOAL_GENESIS_TEST_WRAPPER: COMPLETE"
exit "$STATUS"
