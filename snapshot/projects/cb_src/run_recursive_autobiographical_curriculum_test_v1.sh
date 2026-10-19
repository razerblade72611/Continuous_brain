#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/continuous_brain_v0.12.2}"
cd "$PROJECT_ROOT"

ROOT="state/recursive_autobiographical_curriculum_test_v1"
mkdir -p "$ROOT"
rm -f \
  "$ROOT/result.json" \
  "$ROOT/run.log" \
  "$ROOT/result_bundle.zip" \
  "$ROOT/curriculum_snapshot.pt"

echo "=== v0.36.1 Frozen Recursive Autobiographical Curriculum TEST ==="
echo "READ ONLY. CPU only. No training."
echo "Uses exact v0.36.0 mechanism, thresholds, first-goal horizon, second-goal probe, and frozen-history counterfactual."
echo "Uses sealed TEST subjects 120000-120127 exactly once."
echo "After this run the v0.36 TEST is permanently spent."
echo "No tuning after TEST."

test -s state/recursive_autobiographical_curriculum_v1/result.json
test -s state/endogenous_goal_genesis_test_v1/result.json
test -s state/autobiographical_self_discrepancy_neutral_v1/neutral_autobiographical_self_model.pt
test -s state/recurrent_self_trajectory_v1/recurrent_self_trajectory.pt

python -m py_compile \
  evaluate_recursive_autobiographical_curriculum_test_v1.py \
  test_recursive_autobiographical_curriculum_test_v1.py

python test_recursive_autobiographical_curriculum_test_v1.py

set +e
python evaluate_recursive_autobiographical_curriculum_test_v1.py \
  --state-dir "$ROOT" \
  2>&1 | tee "$ROOT/run.log"
STATUS=${PIPESTATUS[0]}
set -e

if [ "$STATUS" -ne 0 ] && [ "$STATUS" -ne 4 ]; then
  echo "RECURSIVE_AUTOBIOGRAPHICAL_CURRICULUM_TEST_WRAPPER: FAIL_EVALUATOR"
  exit "$STATUS"
fi

grep -q '^RECURSIVE_AUTOBIOGRAPHICAL_CURRICULUM_TEST_COMPLETE$' "$ROOT/run.log"
grep -q '^TRAINING_PERFORMED: false$' "$ROOT/run.log"
grep -q '^THRESHOLDS_CHANGED_FROM_V0360: false$' "$ROOT/run.log"
grep -q '^NEURAL_MODELS_CHANGED: false$' "$ROOT/run.log"
grep -q '^PROTECTED_UNCHANGED: true$' "$ROOT/run.log"
grep -q '^SOURCE_MODELS_UNCHANGED: true$' "$ROOT/run.log"
grep -q '^V036_TEST_SPENT: true$' "$ROOT/run.log"
grep -q '^TEST_TUNING_AFTER_RUN: forbidden$' "$ROOT/run.log"

test -s "$ROOT/result.json"

python - "$ROOT/result.json" "$ROOT/result_bundle.zip" <<'PY'
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import json, sys

r = Path(sys.argv[1])
b = Path(sys.argv[2])
p = json.loads(r.read_text())

assert p["family"] == "v0.36"
assert p["training_performed"] is False
assert p["historical_status"]["v0353"] == "FROZEN_TEST_PASS"
assert p["historical_status"]["v0360"] == "DEV_PASS"
assert p["frozen_mechanism"]["external_second_goal_supplied"] is False
assert p["frozen_mechanism"]["first_goal_horizon"] == 16
assert p["frozen_mechanism"]["second_goal_probe_episodes"] == 4
assert p["frozen_mechanism"]["frozen_bootstrap_counterfactual"] is True
assert p["frozen_mechanism"]["thresholds_changed_from_v0360"] is False
assert p["frozen_mechanism"]["neural_models_changed"] is False
assert p["split"]["test_spent"] is True
assert p["split"]["test_tuning_after_run_forbidden"] is True
assert p["protected_unchanged"] is True
assert p["source_models_unchanged"] is True
assert p["scope"]["open_ended_semantic_goal_space_claimed"] is False
assert p["scope"]["subjective_experience_claimed"] is False
assert p["scope"]["sentience_claimed"] is False
assert p["scope"]["consciousness_claimed"] is False

with ZipFile(b, "w", ZIP_DEFLATED) as z:
    z.write(r, "result.json")
    log = r.with_name("run.log")
    if log.is_file():
        z.write(log, "run.log")
    snap = r.with_name("curriculum_snapshot.pt")
    if snap.is_file():
        z.write(
            snap,
            "curriculum_snapshot.pt",
        )

print(b)
PY

if [ "$STATUS" -eq 0 ]; then
  echo "RECURSIVE_AUTOBIOGRAPHICAL_CURRICULUM_TEST_ADMISSION: PASS_CONFIRMED"
else
  echo "RECURSIVE_AUTOBIOGRAPHICAL_CURRICULUM_TEST_ADMISSION: FAIL_CONFIRMED"
fi

echo "RECURSIVE_AUTOBIOGRAPHICAL_CURRICULUM_TEST_WRAPPER: COMPLETE"
exit "$STATUS"
