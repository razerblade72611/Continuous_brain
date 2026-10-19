#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/continuous_brain_v0.12.2}"
cd "$PROJECT_ROOT"

ROOT="state/event_consistent_self_development_test_v1"
mkdir -p "$ROOT"
rm -f \
  "$ROOT/result.json" \
  "$ROOT/run.log" \
  "$ROOT/result_bundle.zip" \
  "$ROOT/developmental_state_probe.pt" \
  "$ROOT/developmental_state_probe.memory.pt"

echo "=== v0.34.1 Frozen Event-Consistent Self-Development TEST ==="
echo "READ ONLY. CPU only. No training."
echo "Uses exact v0.34.0 controller, thresholds, dynamics, event semantics, and per-subject full-loop definition."
echo "Uses sealed v0.34 TEST subjects 80000-80127."
echo "v0.33.5 remains a frozen TEST FAIL; its spent TEST is never reopened."
echo "After this run the v0.34 TEST is spent and may never be tuned against."

test -s state/event_consistent_self_development_v1/result.json
test -s state/self_development_event_aligned_test_v1/result.json
test -s state/autobiographical_self_discrepancy_neutral_v1/neutral_autobiographical_self_model.pt
test -s state/recurrent_self_trajectory_v1/recurrent_self_trajectory.pt

python -m py_compile \
  evaluate_event_consistent_self_development_test_v1.py \
  test_event_consistent_self_development_test_v1.py

python test_event_consistent_self_development_test_v1.py

set +e
python evaluate_event_consistent_self_development_test_v1.py \
  --state-dir "$ROOT" \
  2>&1 | tee "$ROOT/run.log"
STATUS=${PIPESTATUS[0]}
set -e

if [ "$STATUS" -ne 0 ] && [ "$STATUS" -ne 4 ]; then
  echo "EVENT_CONSISTENT_SELF_DEVELOPMENT_TEST_WRAPPER: FAIL_EVALUATOR"
  exit "$STATUS"
fi

grep -q '^EVENT_CONSISTENT_SELF_DEVELOPMENT_TEST_COMPLETE$' "$ROOT/run.log"
grep -q '^TRAINING_PERFORMED: false$' "$ROOT/run.log"
grep -q '^V0335_FROZEN_TEST_FAILURE_PRESERVED: true$' "$ROOT/run.log"
grep -q '^V033_TEST_REOPENED: false$' "$ROOT/run.log"
grep -q '^PROTECTED_UNCHANGED: true$' "$ROOT/run.log"
grep -q '^SOURCE_MODELS_UNCHANGED: true$' "$ROOT/run.log"
grep -q '^V034_TEST_SPENT: true$' "$ROOT/run.log"
grep -q '^TEST_TUNING_AFTER_RUN: forbidden$' "$ROOT/run.log"

test -s "$ROOT/result.json"

python - "$ROOT/result.json" "$ROOT/result_bundle.zip" <<'PY'
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import json, sys

r = Path(sys.argv[1])
b = Path(sys.argv[2])
p = json.loads(r.read_text())

assert p["family"] == "v0.34"
assert p["training_performed"] is False
assert p["historical_status"]["v0335"] == "FROZEN_TEST_FAIL"
assert p["historical_status"]["v0340"] == "DEV_PASS"
assert p["frozen_mechanism"]["controller_changed"] is False
assert p["frozen_mechanism"]["thresholds_changed"] is False
assert p["frozen_mechanism"]["neural_models_changed"] is False
assert p["frozen_mechanism"]["development_dynamics_changed"] is False
assert p["frozen_mechanism"]["full_loop_definition_changed"] is False
assert p["frozen_mechanism"]["self_estimate_gain_threshold"] == 0.12
assert p["split"]["test_spent"] is True
assert p["split"]["test_tuning_after_run_forbidden"] is True
assert p["protected_unchanged"] is True
assert p["source_models_unchanged"] is True
assert p["v0335_frozen_test_failure_preserved"] is True
assert p["v033_test_reopened"] is False
assert p["scope"]["subjective_experience_claimed"] is False
assert p["scope"]["sentience_claimed"] is False
assert p["scope"]["consciousness_claimed"] is False

with ZipFile(b, "w", ZIP_DEFLATED) as z:
    z.write(r, "result.json")
    log = r.with_name("run.log")
    if log.is_file():
        z.write(log, "run.log")
    state = r.with_name("developmental_state_probe.pt")
    memory = r.with_name("developmental_state_probe.memory.pt")
    if state.is_file():
        z.write(state, "developmental_state_probe.pt")
    if memory.is_file():
        z.write(memory, "developmental_state_probe.memory.pt")

print(b)
PY

if [ "$STATUS" -eq 0 ]; then
  echo "EVENT_CONSISTENT_SELF_DEVELOPMENT_TEST_ADMISSION: PASS_CONFIRMED"
else
  echo "EVENT_CONSISTENT_SELF_DEVELOPMENT_TEST_ADMISSION: FAIL_CONFIRMED"
fi

echo "EVENT_CONSISTENT_SELF_DEVELOPMENT_TEST_WRAPPER: COMPLETE"
exit "$STATUS"
