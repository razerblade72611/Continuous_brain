#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/continuous_brain_v0.12.2}"
cd "$PROJECT_ROOT"

ROOT="state/all_option_success_goal_v1"
mkdir -p "$ROOT"

rm -f \
  "$ROOT/result.json" \
  "$ROOT/run.log" \
  "$ROOT/result_bundle.zip"

echo "=== v0.20.2 All-Option Success-Goal Development Assay ==="
echo "CPU reference backend. No training."
echo "Fresh relative-to-v0.20.1 seeds: 14001-14040."

set +e
python evaluate_all_option_success_goal_dev_v1.py \
  --output "$ROOT/result.json" \
  2>&1 | tee "$ROOT/run.log"
STATUS=${PIPESTATUS[0]}
set -e

# A scientific gate failure (4) is a valid completed experiment.
if [ "$STATUS" -ne 0 ] && [ "$STATUS" -ne 4 ]; then
  echo "ALL_OPTION_SUCCESS_GOAL_WRAPPER: FAIL_EVALUATOR"
  exit "$STATUS"
fi

grep -q '^ALL_OPTION_SUCCESS_GOAL_DEV_COMPLETE$' "$ROOT/run.log"
grep -q '^PROTECTED_UNCHANGED: true$' "$ROOT/run.log"
grep -q '^TRAINING_PERFORMED: false$' "$ROOT/run.log"
grep -q '^V019_REMAINS_CLOSED: true$' "$ROOT/run.log"
test -s "$ROOT/result.json"

python - "$ROOT/result.json" "$ROOT/result_bundle.zip" <<'PY'
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import json,sys
r=Path(sys.argv[1]);b=Path(sys.argv[2]);p=json.loads(r.read_text())
assert p["training_performed"] is False
assert p["protected_unchanged"] is True
assert p["v019_remains_closed"] is True
with ZipFile(b,"w",ZIP_DEFLATED) as z:
    z.write(r,"result.json")
    log=r.with_name("run.log")
    if log.is_file():z.write(log,"run.log")
print(b)
PY

if [ "$STATUS" -eq 0 ]; then
  echo "ALL_OPTION_SUCCESS_GOAL_DEV_GATE: PASS_CONFIRMED"
else
  echo "ALL_OPTION_SUCCESS_GOAL_DEV_GATE: FAIL_CONFIRMED"
fi
echo "ALL_OPTION_SUCCESS_GOAL_WRAPPER: COMPLETE"
exit "$STATUS"
