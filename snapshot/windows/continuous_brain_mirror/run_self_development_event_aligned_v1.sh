#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/continuous_brain_v0.12.2}"
cd "$PROJECT_ROOT"

ROOT="state/self_development_event_aligned_v1"
mkdir -p "$ROOT"
rm -f \
  "$ROOT/result.json" \
  "$ROOT/run.log" \
  "$ROOT/result_bundle.zip" \
  "$ROOT/developmental_state_probe.pt" \
  "$ROOT/developmental_state_probe.memory.pt"

echo "=== v0.33.4 Event-Aligned Self-Development DEV Repair ==="
echo "Source v0.33.0 and v0.33.1 remain formal FAIL."
echo "No neural training. No controller change. No threshold change."
echo "Replaces only the final-state self-estimate gate with the causally aligned decision-event gate justified by v0.33.3."
echo "Reuses DEV subjects 10000-10127."
echo "TEST subjects 20000-20127 remain sealed."

test -s state/self_development_loop_v1/result.json
test -s state/self_development_hysteresis_v1/result.json
test -s state/self_development_gain_timing_diagnostic_v1/result.json
test -s state/self_development_event_order_diagnostic_v1/result.json
test -s state/autobiographical_self_discrepancy_neutral_v1/neutral_autobiographical_self_model.pt
test -s state/recurrent_self_trajectory_v1/recurrent_self_trajectory.pt

python -m py_compile \
  evaluate_self_development_event_aligned_v1.py \
  test_self_development_event_aligned_v1.py

python test_self_development_event_aligned_v1.py

set +e
python evaluate_self_development_event_aligned_v1.py \
  --state-dir "$ROOT" \
  2>&1 | tee "$ROOT/run.log"
STATUS=${PIPESTATUS[0]}
set -e

if [ "$STATUS" -ne 0 ] && [ "$STATUS" -ne 4 ]; then
  echo "SELF_DEVELOPMENT_EVENT_ALIGNED_WRAPPER: FAIL_EVALUATOR"
  exit "$STATUS"
fi

grep -q '^SELF_DEVELOPMENT_EVENT_ALIGNED_RUN_COMPLETE$' "$ROOT/run.log"
grep -q '^TRAINING_PERFORMED: false$' "$ROOT/run.log"
grep -q '^CONTROLLER_CHANGED: false$' "$ROOT/run.log"
grep -q '^THRESHOLD_CHANGED: false$' "$ROOT/run.log"
grep -q '^PROTECTED_UNCHANGED: true$' "$ROOT/run.log"
grep -q '^DEVELOPMENT_TEST_TOUCHED: false$' "$ROOT/run.log"

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
assert p["repair"]["controller_changed"] is False
assert p["repair"]["development_dynamics_changed"] is False
assert p["repair"]["neural_models_changed"] is False
assert p["repair"]["threshold_changed"] is False
assert p["repair"]["self_estimate_gain_threshold"] == 0.12
assert p["protected_unchanged"] is True
assert p["development_test_touched"] is False

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
  echo "SELF_DEVELOPMENT_EVENT_ALIGNED_DEV_GATE: PASS_CONFIRMED"
else
  echo "SELF_DEVELOPMENT_EVENT_ALIGNED_DEV_GATE: FAIL_CONFIRMED"
fi

echo "SELF_DEVELOPMENT_EVENT_ALIGNED_WRAPPER: COMPLETE"
exit "$STATUS"
