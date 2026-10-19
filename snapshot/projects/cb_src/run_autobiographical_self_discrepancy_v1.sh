#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/continuous_brain_v0.12.2}"
cd "$PROJECT_ROOT"

ROOT="state/autobiographical_self_discrepancy_v1"
mkdir -p "$ROOT"
rm -f \
  "$ROOT/result.json" \
  "$ROOT/run.log" \
  "$ROOT/result_bundle.zip" \
  "$ROOT/autobiographical_self_model.pt" \
  "$ROOT/autobiographical_memory_probe.pt"

echo "=== v0.32.0 Autobiographical Expected-Self / Actual-Self Discrepancy ==="
echo "Cognition milestone. No robot training."
echo "Uses frozen v0.31 recurrent self-trajectory model to form expectations."
echo "Stores expected trajectory before action and actual trajectory afterward."
echo "Learns only from chronological autobiographical discrepancy history."
echo "Autobiographical TEST subjects 6000-6255 remain sealed."

test -s state/recurrent_self_trajectory_v1/recurrent_self_trajectory.pt
test -s state/recurrent_self_trajectory_v1/result.json
test -s state/recurrent_self_trajectory_test_v1/result.json

python -m py_compile \
  continuous_brain/autobiographical_self_discrepancy_v1.py \
  train_evaluate_autobiographical_self_discrepancy_v1.py \
  test_autobiographical_self_discrepancy_v1.py

python test_autobiographical_self_discrepancy_v1.py

set +e
python train_evaluate_autobiographical_self_discrepancy_v1.py \
  --state-dir "$ROOT" \
  2>&1 | tee "$ROOT/run.log"
STATUS=${PIPESTATUS[0]}
set -e

if [ "$STATUS" -ne 0 ] && [ "$STATUS" -ne 4 ]; then
  echo "AUTOBIOGRAPHICAL_SELF_DISCREPANCY_WRAPPER: FAIL_EVALUATOR"
  exit "$STATUS"
fi

grep -q '^AUTOBIOGRAPHICAL_SELF_DISCREPANCY_RUN_COMPLETE$' "$ROOT/run.log"
grep -q '^PROTECTED_UNCHANGED: true$' "$ROOT/run.log"
grep -q '^V0311_TEST_PASS_PRESERVED: true$' "$ROOT/run.log"
grep -q '^AUTOBIO_TEST_TOUCHED: false$' "$ROOT/run.log"

test -s "$ROOT/result.json"
test -s "$ROOT/autobiographical_self_model.pt"

python - \
  "$ROOT/result.json" \
  "$ROOT/autobiographical_self_model.pt" \
  "$ROOT/result_bundle.zip" <<'PY'
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import json,sys

r=Path(sys.argv[1])
m=Path(sys.argv[2])
b=Path(sys.argv[3])
p=json.loads(r.read_text())

assert p["protected_unchanged"] is True
assert p["v0311_test_pass_preserved"] is True
assert p["autobio_test_touched"] is False
assert p["training"]["autobiographical_self_model"] is True
assert p["training"]["recurrent_trajectory_model"] is False
assert p["architecture"]["identity_input"] is False
assert p["architecture"]["current_world_input"] is False
assert p["architecture"]["current_self_input"] is False
assert p["scope"]["subjective_experience_claimed"] is False
assert p["scope"]["sentience_claimed"] is False
assert p["scope"]["consciousness_claimed"] is False

with ZipFile(b,"w",ZIP_DEFLATED) as z:
    z.write(r,"result.json")
    z.write(m,"autobiographical_self_model.pt")
    mem=r.with_name("autobiographical_memory_probe.pt")
    if mem.is_file():
        z.write(mem,"autobiographical_memory_probe.pt")
    log=r.with_name("run.log")
    if log.is_file():
        z.write(log,"run.log")
print(b)
PY

if [ "$STATUS" -eq 0 ]; then
  echo "AUTOBIOGRAPHICAL_SELF_DISCREPANCY_DEV_GATE: PASS_CONFIRMED"
else
  echo "AUTOBIOGRAPHICAL_SELF_DISCREPANCY_DEV_GATE: FAIL_CONFIRMED"
fi

echo "AUTOBIOGRAPHICAL_SELF_DISCREPANCY_WRAPPER: COMPLETE"
exit "$STATUS"
