#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/continuous_brain_v0.12.2}"
cd "$PROJECT_ROOT"

ROOT="state/endogenous_goal_completion_diagnostic_v1"
mkdir -p "$ROOT"
rm -f "$ROOT/result.json" "$ROOT/run.log" "$ROOT/result_bundle.zip"

echo "=== v0.35.1 Endogenous Goal Completion Dynamics Diagnostic ==="
echo "READ ONLY. NO TRAINING."
echo "Preserves v0.35.0 as a formal DEV FAIL."
echo "Reuses spent DEV subjects 90000-90127."
echo "Traces exact self-authored goal completion dynamics through 32 target episodes."
echo "TEST subjects 100000-100127 remain sealed."

test -s state/endogenous_goal_genesis_v1/result.json
test -s state/event_consistent_self_development_test_v1/result.json
test -s state/autobiographical_self_discrepancy_neutral_v1/neutral_autobiographical_self_model.pt
test -s state/recurrent_self_trajectory_v1/recurrent_self_trajectory.pt

python -m py_compile diagnose_endogenous_goal_completion_v1.py

python diagnose_endogenous_goal_completion_v1.py \
  --state-dir "$ROOT" \
  2>&1 | tee "$ROOT/run.log"

grep -q '^ENDOGENOUS_GOAL_COMPLETION_DIAGNOSTIC_COMPLETE$' "$ROOT/run.log"
grep -q '^PROTECTED_UNCHANGED: true$' "$ROOT/run.log"
grep -q '^V035_TEST_TOUCHED: false$' "$ROOT/run.log"

python - "$ROOT/result.json" "$ROOT/result_bundle.zip" <<'PY'
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import json, sys

r = Path(sys.argv[1])
b = Path(sys.argv[2])
p = json.loads(r.read_text())

assert p["training_performed"] is False
assert p["read_only_diagnostic"] is True
assert p["source_v0350_formal_status"] == "FAIL"
assert p["protected_unchanged"] is True
assert p["v035_test_touched"] is False

with ZipFile(b, "w", ZIP_DEFLATED) as z:
    z.write(r, "result.json")
    log = r.with_name("run.log")
    if log.is_file():
        z.write(log, "run.log")

print(b)
PY

echo "ENDOGENOUS_GOAL_COMPLETION_DIAGNOSTIC_WRAPPER: COMPLETE"
