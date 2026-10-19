#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/continuous_brain_v0.12.2}"
cd "$PROJECT_ROOT"

ROOT="state/self_development_gain_timing_diagnostic_v1"
mkdir -p "$ROOT"
rm -f "$ROOT/result.json" "$ROOT/run.log" "$ROOT/result_bundle.zip"

echo "=== v0.33.2 Self-Estimate Gain Timing Diagnostic ==="
echo "READ ONLY. NO TRAINING."
echo "Reuses spent v0.33 DEV subjects 10000-10127."
echo "Diagnoses whether the lone v0.33.1 failure is a final-vs-peak measurement mismatch."
echo "TEST subjects 20000-20127 remain sealed."

test -s state/self_development_hysteresis_v1/result.json
test -s state/self_development_loop_v1/result.json
test -s state/autobiographical_self_discrepancy_neutral_v1/neutral_autobiographical_self_model.pt
test -s state/recurrent_self_trajectory_v1/recurrent_self_trajectory.pt

python -m py_compile diagnose_self_development_gain_timing_v1.py

python diagnose_self_development_gain_timing_v1.py \
  --state-dir "$ROOT" \
  2>&1 | tee "$ROOT/run.log"

grep -q '^SELF_DEVELOPMENT_GAIN_TIMING_DIAGNOSTIC_COMPLETE$' "$ROOT/run.log"
grep -q '^PROTECTED_UNCHANGED: true$' "$ROOT/run.log"
grep -q '^DEVELOPMENT_TEST_TOUCHED: false$' "$ROOT/run.log"

python - "$ROOT/result.json" "$ROOT/result_bundle.zip" <<'PY'
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import json, sys

r = Path(sys.argv[1])
b = Path(sys.argv[2])
p = json.loads(r.read_text())

assert p["training_performed"] is False
assert p["read_only_diagnostic"] is True
assert p["source_v0331_formal_status"] == "FAIL"
assert p["protected_unchanged"] is True
assert p["development_test_touched"] is False

with ZipFile(b, "w", ZIP_DEFLATED) as z:
    z.write(r, "result.json")
    log = r.with_name("run.log")
    if log.is_file():
        z.write(log, "run.log")

print(b)
PY

echo "SELF_DEVELOPMENT_GAIN_TIMING_DIAGNOSTIC_WRAPPER: COMPLETE"
