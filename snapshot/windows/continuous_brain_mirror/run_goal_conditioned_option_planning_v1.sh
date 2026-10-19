#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/continuous_brain_v0.12.2}"
cd "$PROJECT_ROOT"

ROOT="state/goal_conditioned_option_planning_v1"
mkdir -p "$ROOT"

rm -f \
  "$ROOT/result.json" \
  "$ROOT/run.log" \
  "$ROOT/result_bundle.zip"

echo "=== v0.20.0 Goal-Conditioned Autonomous Option Planning ==="
echo "CPU reference backend only."
echo "No training. No model checkpoint writes."

python evaluate_goal_conditioned_option_planning_v1.py \
  --output "$ROOT/result.json" \
  2>&1 | tee "$ROOT/run.log"

grep -q '^PROTECTED_UNCHANGED: true$' "$ROOT/run.log"
grep -q '^TRAINING_PERFORMED: false$' "$ROOT/run.log"
grep -q '^AUTONOMOUS_ENVIRONMENT_CONTROL: true$' "$ROOT/run.log"
grep -q '^TEACHER_DRIVEN_TRAJECTORIES: false$' "$ROOT/run.log"
grep -q '^V019_REMAINS_CLOSED: true$' "$ROOT/run.log"
grep -q '^GOAL_CONDITIONED_OPTION_PLANNING_RUN: PASS$' "$ROOT/run.log"

test -s "$ROOT/result.json"

python - "$ROOT/result.json" "$ROOT/result_bundle.zip" <<'PY'
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import json, sys

r = Path(sys.argv[1])
b = Path(sys.argv[2])
p = json.loads(r.read_text())

assert p["software_status"] == "PASS"
assert p["training_performed"] is False
assert p["autonomous_environment_control"] is True
assert p["teacher_driven_trajectories"] is False
assert p["protected_unchanged"] is True
assert p["scientific_admission_decided"] is False

with ZipFile(b, "w", ZIP_DEFLATED) as z:
    z.write(r, "result.json")
    log = r.with_name("run.log")
    if log.is_file():
        z.write(log, "run.log")
print(b)
PY

echo "GOAL_CONDITIONED_OPTION_PLANNING_WRAPPER: PASS"
