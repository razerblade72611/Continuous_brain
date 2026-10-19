#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/continuous_brain_v0.12.2}"
cd "$PROJECT_ROOT"

ROOT="state/self_broadcast_value_priority_gradient_safe_v1"
mkdir -p "$ROOT"
rm -f \
  "$ROOT/result.json" \
  "$ROOT/run.log" \
  "$ROOT/result_bundle.zip" \
  "$ROOT/value_priority_gradient_safe.pt"

echo "=== v0.29.2 Gradient-Safe Value-Priority Repair ==="
echo "Repairs ONLY the failed priority readout."
echo "Shared v0.29.0 64-D self broadcast stays frozen."
echo "v0.29.0 remains FAIL; v0.29.1 remains FAIL."
echo "Reuses spent DEV 20000-22999. TEST 30000-32999 remains sealed."

test -s state/self_broadcast_centrality_v1/result.json
test -s state/self_broadcast_centrality_v1/self_broadcast_hub.pt
test -s state/self_broadcast_value_priority_repair_v1/result.json
test -s state/persistent_self_autobiographical_test_v1/result.json

python -m py_compile \
  continuous_brain/self_broadcast_value_priority_gradient_safe_v1.py \
  train_evaluate_self_broadcast_value_priority_gradient_safe_v1.py \
  test_self_broadcast_value_priority_gradient_safe_v1.py

python test_self_broadcast_value_priority_gradient_safe_v1.py

set +e
python train_evaluate_self_broadcast_value_priority_gradient_safe_v1.py \
  --state-dir "$ROOT" \
  2>&1 | tee "$ROOT/run.log"
STATUS=${PIPESTATUS[0]}
set -e

if [ "$STATUS" -ne 0 ] && [ "$STATUS" -ne 4 ]; then
  echo "SELF_BROADCAST_VALUE_PRIORITY_GRADIENT_SAFE_WRAPPER: FAIL_EVALUATOR"
  exit "$STATUS"
fi

grep -q '^SELF_BROADCAST_VALUE_PRIORITY_GRADIENT_SAFE_COMPLETE$' "$ROOT/run.log"
grep -q '^PROTECTED_UNCHANGED: true$' "$ROOT/run.log"
grep -q '^SOURCE_HUB_UNCHANGED: true$' "$ROOT/run.log"
grep -q '^BROADCAST_TEST_TOUCHED: false$' "$ROOT/run.log"

test -s "$ROOT/result.json"
test -s "$ROOT/value_priority_gradient_safe.pt"

python - \
  "$ROOT/result.json" \
  "$ROOT/value_priority_gradient_safe.pt" \
  "$ROOT/result_bundle.zip" <<'PY'
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import json, sys

r = Path(sys.argv[1])
h = Path(sys.argv[2])
b = Path(sys.argv[3])
p = json.loads(r.read_text())

assert p["source_v0290_formal_status"] == "FAIL"
assert p["v0291_repair_formal_status"] == "FAIL"
assert p["source_hub_unchanged"] is True
assert p["protected_unchanged"] is True
assert p["broadcast_test_touched"] is False
assert p["training"]["priority_repair_head"] is True
assert p["training"]["shared_self_broadcast_hub"] is False

with ZipFile(b, "w", ZIP_DEFLATED) as z:
    z.write(r, "result.json")
    z.write(h, "value_priority_gradient_safe.pt")
    log = r.with_name("run.log")
    if log.is_file():
        z.write(log, "run.log")
print(b)
PY

if [ "$STATUS" -eq 0 ]; then
  echo "SELF_BROADCAST_VALUE_PRIORITY_GRADIENT_SAFE_DEV_GATE: PASS_CONFIRMED"
else
  echo "SELF_BROADCAST_VALUE_PRIORITY_GRADIENT_SAFE_DEV_GATE: FAIL_CONFIRMED"
fi

echo "SELF_BROADCAST_VALUE_PRIORITY_GRADIENT_SAFE_WRAPPER: COMPLETE"
exit "$STATUS"
