#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/continuous_brain_v0.12.2}"
cd "$PROJECT_ROOT"

ROOT="state/self_development_event_aligned_test_v1"
mkdir -p "$ROOT"
rm -f \
  "$ROOT/result.json" \
  "$ROOT/run.log" \
  "$ROOT/result_bundle.zip" \
  "$ROOT/developmental_state_probe.pt" \
  "$ROOT/developmental_state_probe.memory.pt"

echo "=== v0.33.5 Frozen Event-Aligned Self-Development TEST ==="
echo "READ ONLY. CPU only. No training."
echo "Uses exact v0.33.4 controller, dynamics, thresholds, and event-aligned gate."
echo "Uses sealed developmental TEST subjects 20000-20127 exactly once."
echo "After this run the v0.33 developmental TEST is spent and may never be tuned against."

test -s state/self_development_event_aligned_v1/result.json
test -s state/self_development_hysteresis_v1/result.json
test -s state/self_development_event_order_diagnostic_v1/result.json
test -s state/autobiographical_self_discrepancy_neutral_v1/neutral_autobiographical_self_model.pt
test -s state/recurrent_self_trajectory_v1/recurrent_self_trajectory.pt

python -m py_compile \
  evaluate_self_development_event_aligned_test_v1.py \
  test_self_development_event_aligned_test_v1.py

python test_self_development_event_aligned_test_v1.py

set +e
python evaluate_self_development_event_aligned_test_v1.py \
  --state-dir "$ROOT" \
  2>&1 | tee "$ROOT/run.log"
STATUS=${PIPESTATUS[0]}
set -e

if [ "$STATUS" -ne 0 ] && [ "$STATUS" -ne 4 ]; then
  echo "SELF_DEVELOPMENT_EVENT_ALIGNED_TEST_WRAPPER: FAIL_EVALUATOR"
  exit "$STATUS"
fi

grep -q '^SELF_DEVELOPMENT_EVENT_ALIGNED_TEST_COMPLETE$' "$ROOT/run.log"
grep -q '^TRAINING_PERFORMED: false$' "$ROOT/run.log"
grep -q '^CONTROLLER_CHANGED: false$' "$ROOT/run.log"
grep -q '^THRESHOLD_CHANGED: false$' "$ROOT/run.log"
grep -q '^PROTECTED_UNCHANGED: true$' "$ROOT/run.log"
grep -q '^SOURCE_MODELS_UNCHANGED: true$' "$ROOT/run.log"
grep -q '^DEVELOPMENT_TEST_SPENT: true$' "$ROOT/run.log"
grep -q '^TEST_TUNING_AFTER_RUN: forbidden$' "$ROOT/run.log"

test -s "$ROOT/result.json"

python - "$ROOT/result.json" "$ROOT/result_bundle.zip" <<'PY'
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import json, sys

r = Path(sys.argv[1])
b = Path(sys.argv[2])
p = json.loads(r.read_text())

assert p["training_performed"] is False
assert p["historical_status"]["v0330"] == "FAIL"
assert p["historical_status"]["v0331"] == "FAIL"
assert p["historical_status"]["v0334"] == "DEV_PASS"
assert p["frozen_mechanism"]["controller_changed"] is False
assert p["frozen_mechanism"]["threshold_changed"] is False
assert p["frozen_mechanism"]["development_dynamics_changed"] is False
assert p["frozen_mechanism"]["neural_models_changed"] is False
assert p["frozen_mechanism"]["self_estimate_gain_threshold"] == 0.12
assert p["split"]["test_spent"] is True
assert p["split"]["test_tuning_after_run_forbidden"] is True
assert p["protected_unchanged"] is True
assert p["source_models_unchanged"] is True
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
  echo "SELF_DEVELOPMENT_EVENT_ALIGNED_TEST_ADMISSION: PASS_CONFIRMED"
else
  echo "SELF_DEVELOPMENT_EVENT_ALIGNED_TEST_ADMISSION: FAIL_CONFIRMED"
fi

echo "SELF_DEVELOPMENT_EVENT_ALIGNED_TEST_WRAPPER: COMPLETE"
exit "$STATUS"
