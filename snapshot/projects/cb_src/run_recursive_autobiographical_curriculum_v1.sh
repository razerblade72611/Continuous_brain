#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/continuous_brain_v0.12.2}"
cd "$PROJECT_ROOT"

ROOT="state/recursive_autobiographical_curriculum_v1"
mkdir -p "$ROOT"
rm -f \
  "$ROOT/result.json" \
  "$ROOT/run.log" \
  "$ROOT/result_bundle.zip" \
  "$ROOT/curriculum_snapshot.pt"

echo "=== v0.36.0 Recursive Autobiographical Curriculum ==="
echo "Cognition milestone. No neural training."
echo "Requires frozen v0.35.3 TEST PASS."
echo "No external second developmental goal is supplied."
echo "After the first self-authored goal completes, updated autobiography must author what comes next."
echo "Frozen-bootstrap autobiography is the counterfactual: it should reproduce the old goal rather than advance the curriculum."
echo "Fresh DEV subjects 110000-110127."
echo "Fresh TEST subjects 120000-120127 remain sealed."

test -s state/endogenous_goal_genesis_test_v1/result.json
test -s state/autobiographical_self_discrepancy_neutral_v1/neutral_autobiographical_self_model.pt
test -s state/recurrent_self_trajectory_v1/recurrent_self_trajectory.pt

python -m py_compile \
  continuous_brain/recursive_autobiographical_curriculum_v1.py \
  evaluate_recursive_autobiographical_curriculum_v1.py \
  test_recursive_autobiographical_curriculum_v1.py

python test_recursive_autobiographical_curriculum_v1.py

set +e
python evaluate_recursive_autobiographical_curriculum_v1.py \
  --state-dir "$ROOT" \
  2>&1 | tee "$ROOT/run.log"
STATUS=${PIPESTATUS[0]}
set -e

if [ "$STATUS" -ne 0 ] && [ "$STATUS" -ne 4 ]; then
  echo "RECURSIVE_AUTOBIOGRAPHICAL_CURRICULUM_WRAPPER: FAIL_EVALUATOR"
  exit "$STATUS"
fi

grep -q '^RECURSIVE_AUTOBIOGRAPHICAL_CURRICULUM_RUN_COMPLETE$' "$ROOT/run.log"
grep -q '^TRAINING_PERFORMED: false$' "$ROOT/run.log"
grep -q '^PROTECTED_UNCHANGED: true$' "$ROOT/run.log"
grep -q '^V0353_TEST_PASS_PRESERVED: true$' "$ROOT/run.log"
grep -q '^V036_TEST_TOUCHED: false$' "$ROOT/run.log"

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
assert p["source_v0353_test_status"] == "PASS"
assert p["architecture"]["external_second_goal_supplied"] is False
assert p["architecture"]["frozen_bootstrap_counterfactual"] is True
assert p["split"]["test_touched"] is False
assert p["protected_unchanged"] is True
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
        z.write(snap, "curriculum_snapshot.pt")

print(b)
PY

if [ "$STATUS" -eq 0 ]; then
  echo "RECURSIVE_AUTOBIOGRAPHICAL_CURRICULUM_DEV_GATE: PASS_CONFIRMED"
else
  echo "RECURSIVE_AUTOBIOGRAPHICAL_CURRICULUM_DEV_GATE: FAIL_CONFIRMED"
fi

echo "RECURSIVE_AUTOBIOGRAPHICAL_CURRICULUM_WRAPPER: COMPLETE"
exit "$STATUS"
