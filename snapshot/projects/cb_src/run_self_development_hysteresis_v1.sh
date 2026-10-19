#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/continuous_brain_v0.12.2}"
cd "$PROJECT_ROOT"

ROOT="state/self_development_hysteresis_v1"
mkdir -p "$ROOT"
rm -f \
  "$ROOT/result.json" \
  "$ROOT/run.log" \
  "$ROOT/result_bundle.zip" \
  "$ROOT/developmental_state_probe.pt" \
  "$ROOT/developmental_state_probe.memory.pt"

echo "=== v0.33.1 History-Relative Developmental Hysteresis Repair ==="
echo "Source v0.33.0 remains formal DEV FAIL."
echo "No neural training."
echo "Replaces only the failed absolute 0.50 readiness threshold."
echo "Graduation uses the already-preregistered +0.12 autobiographical self-estimate gain."
echo "Adds frozen-autobiography control: capability may improve, but behavior must not graduate without updated history."
echo "Reuses DEV subjects 10000-10127. TEST subjects 20000-20127 remain sealed."

test -s state/self_development_loop_v1/result.json
test -s state/recurrent_self_trajectory_v1/recurrent_self_trajectory.pt
test -s state/autobiographical_self_discrepancy_neutral_v1/neutral_autobiographical_self_model.pt
test -s state/autobiographical_self_discrepancy_neutral_test_v1/result.json

python -m py_compile \
  continuous_brain/self_development_hysteresis_v1.py \
  evaluate_self_development_hysteresis_v1.py \
  test_self_development_hysteresis_v1.py

python test_self_development_hysteresis_v1.py

set +e
python evaluate_self_development_hysteresis_v1.py \
  --state-dir "$ROOT" \
  2>&1 | tee "$ROOT/run.log"
STATUS=${PIPESTATUS[0]}
set -e

if [ "$STATUS" -ne 0 ] && [ "$STATUS" -ne 4 ]; then
  echo "SELF_DEVELOPMENT_HYSTERESIS_WRAPPER: FAIL_EVALUATOR"
  exit "$STATUS"
fi

grep -q '^SELF_DEVELOPMENT_HYSTERESIS_RUN_COMPLETE$' "$ROOT/run.log"
grep -q '^TRAINING_PERFORMED: false$' "$ROOT/run.log"
grep -q '^PROTECTED_UNCHANGED: true$' "$ROOT/run.log"
grep -q '^V0322_TEST_PASS_PRESERVED: true$' "$ROOT/run.log"
grep -q '^DEVELOPMENT_TEST_TOUCHED: false$' "$ROOT/run.log"

test -s "$ROOT/result.json"

python - "$ROOT/result.json" "$ROOT/result_bundle.zip" <<'PY'
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import json, sys

r = Path(sys.argv[1])
b = Path(sys.argv[2])
p = json.loads(r.read_text())

assert p["source_v0330_formal_status"] == "FAIL"
assert p["training_performed"] is False
assert p["repair"]["baseline_relative"] is True
assert p["repair"]["one_way_hysteresis"] is True
assert p["repair"]["graduation_gain"] == 0.12
assert p["repair"]["hidden_capability_used_for_decision"] is False
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
  echo "SELF_DEVELOPMENT_HYSTERESIS_DEV_GATE: PASS_CONFIRMED"
else
  echo "SELF_DEVELOPMENT_HYSTERESIS_DEV_GATE: FAIL_CONFIRMED"
fi

echo "SELF_DEVELOPMENT_HYSTERESIS_WRAPPER: COMPLETE"
exit "$STATUS"
