#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/continuous_brain_v0.12.2}"
cd "$PROJECT_ROOT"

ROOT="state/persistent_self_autobiographical_v1"
mkdir -p "$ROOT"
rm -f \
  "$ROOT/result.json" \
  "$ROOT/run.log" \
  "$ROOT/result_bundle.zip" \
  "$ROOT/agency_self_model.pt" \
  "$ROOT/persistent_self_seed.pt" \
  "$ROOT/continuity_probe.pt"

echo "=== v0.28.0 Persistent Self + Autobiographical Continuity ==="
echo "Cognition milestone. Not a robot-controller milestone."
echo "Trains only the additive agency/self-transition model."
echo "Preserves InfantBrain, admitted memories, planner, THOR results, and v0.27 FAIL."
echo "Self TEST 30000-33999 remains sealed."

test -s state/child_v113_dml_smoke/infant_training.pt
test -s state/working_memory_recall_capacity_control_phase1_v1/latest.portable.pt
test -s state/multi_item_episodic_memory_v1/episodic_reader.pt
test -s state/memory_guided_goal_option_v1/goal_decoder.pt
test -s state/consequence_replanning_v1/consequence_model.pt
test -s state/consequence_replanning_test_v1/result.json
test -s state/thor_discovery_embodiment_admission_v1/result.json
test -s state/thor_episodic_door_replanning_v1/result.json

python -m py_compile \
  continuous_brain/persistent_self_autobiographical_v1.py \
  train_evaluate_persistent_self_autobiographical_v1.py \
  test_persistent_self_autobiographical_v1.py

python test_persistent_self_autobiographical_v1.py

set +e
python train_evaluate_persistent_self_autobiographical_v1.py \
  --state-dir "$ROOT" \
  2>&1 | tee "$ROOT/run.log"
STATUS=${PIPESTATUS[0]}
set -e

if [ "$STATUS" -ne 0 ] && [ "$STATUS" -ne 4 ]; then
  echo "PERSISTENT_SELF_AUTOBIOGRAPHICAL_WRAPPER: FAIL_EVALUATOR"
  exit "$STATUS"
fi

grep -q '^PERSISTENT_SELF_AUTOBIOGRAPHICAL_RUN_COMPLETE$' "$ROOT/run.log"
grep -q '^PROTECTED_UNCHANGED: true$' "$ROOT/run.log"
grep -q '^V027_FORMAL_STATUS_PRESERVED: FAIL$' "$ROOT/run.log"
grep -q '^V027_MOTOR_POLISH_DEFERRED: true$' "$ROOT/run.log"
grep -q '^SELF_TEST_TOUCHED: false$' "$ROOT/run.log"

test -s "$ROOT/result.json"
test -s "$ROOT/agency_self_model.pt"
test -s "$ROOT/persistent_self_seed.pt"

python - \
  "$ROOT/result.json" \
  "$ROOT/agency_self_model.pt" \
  "$ROOT/persistent_self_seed.pt" \
  "$ROOT/result_bundle.zip" <<'PY'
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import json, sys

r = Path(sys.argv[1])
m = Path(sys.argv[2])
s = Path(sys.argv[3])
b = Path(sys.argv[4])

p = json.loads(r.read_text())
assert p["protected_unchanged"] is True
assert p["split"]["test_touched"] is False
assert p["v027_formal_status_preserved"] == "FAIL"
assert p["v027_motor_polish_deferred"] is True
assert p["training"]["agency_self_model"] is True
assert p["training"]["historical_brain"] is False
assert p["scope"]["robot_controller_milestone"] is False

with ZipFile(b, "w", ZIP_DEFLATED) as z:
    z.write(r, "result.json")
    z.write(m, "agency_self_model.pt")
    z.write(s, "persistent_self_seed.pt")
    log = r.with_name("run.log")
    if log.is_file():
        z.write(log, "run.log")

print(b)
PY

if [ "$STATUS" -eq 0 ]; then
  echo "PERSISTENT_SELF_AUTOBIOGRAPHICAL_DEV_GATE: PASS_CONFIRMED"
else
  echo "PERSISTENT_SELF_AUTOBIOGRAPHICAL_DEV_GATE: FAIL_CONFIRMED"
fi

echo "PERSISTENT_SELF_AUTOBIOGRAPHICAL_WRAPPER: COMPLETE"
exit "$STATUS"
