#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/continuous_brain_v0.12.2}"
cd "$PROJECT_ROOT"

ROOT="state/same_option_bootstrap_fresh_validation_v1_1"
mkdir -p "$ROOT"

rm -f "$ROOT/result.json" "$ROOT/run.log" "$ROOT/result_bundle.zip"

echo "=== v0.20.6a Same-Option Bootstrap Fresh Validation (software patch) ==="
echo "CPU only. No training."
echo "Fresh relative-to-repair seeds: 15001-15100."
echo "After this run, Detour planner tuning stops regardless of outcome."

set +e
python validate_same_option_bootstrap_fresh_v1_1.py \
  --output "$ROOT/result.json" \
  2>&1 | tee "$ROOT/run.log"
STATUS=${PIPESTATUS[0]}
set -e

if [ "$STATUS" -ne 0 ] && [ "$STATUS" -ne 4 ]; then
  echo "SAME_OPTION_BOOTSTRAP_FRESH_WRAPPER: FAIL_EVALUATOR"
  exit "$STATUS"
fi

grep -q '^SAME_OPTION_BOOTSTRAP_FRESH_VALIDATION_COMPLETE$' "$ROOT/run.log"
grep -q '^PROTECTED_UNCHANGED: true$' "$ROOT/run.log"
grep -q '^TRAINING_PERFORMED: false$' "$ROOT/run.log"
grep -q '^V019_REMAINS_CLOSED: true$' "$ROOT/run.log"
grep -q '^DETOUR_PLANNER_TUNING_AFTER_THIS_RUN: forbidden$' "$ROOT/run.log"
test -s "$ROOT/result.json"

python - "$ROOT/result.json" "$ROOT/result_bundle.zip" <<'PY'
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import json, sys

r=Path(sys.argv[1]); b=Path(sys.argv[2])
p=json.loads(r.read_text())

assert p["training_performed"] is False
assert p["protected_unchanged"] is True
assert p["v019_remains_closed"] is True

with ZipFile(b,"w",ZIP_DEFLATED) as z:
    z.write(r,"result.json")
    log=r.with_name("run.log")
    if log.is_file():
        z.write(log,"run.log")
print(b)
PY

if [ "$STATUS" -eq 0 ]; then
  echo "SAME_OPTION_BOOTSTRAP_FRESH_VALIDATION: PASS_CONFIRMED"
else
  echo "SAME_OPTION_BOOTSTRAP_FRESH_VALIDATION: FAIL_CONFIRMED"
fi

echo "SAME_OPTION_BOOTSTRAP_FRESH_WRAPPER: COMPLETE"
exit "$STATUS"
