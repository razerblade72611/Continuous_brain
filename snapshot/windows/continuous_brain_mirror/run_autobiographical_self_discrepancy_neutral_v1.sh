#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/continuous_brain_v0.12.2}"
cd "$PROJECT_ROOT"

ROOT="state/autobiographical_self_discrepancy_neutral_v1"
mkdir -p "$ROOT"
rm -f \
  "$ROOT/result.json" \
  "$ROOT/run.log" \
  "$ROOT/result_bundle.zip" \
  "$ROOT/neutral_autobiographical_self_model.pt"

echo "=== v0.32.1 Structurally Neutral Autobiographical Discrepancy Repair ==="
echo "Source v0.32.0 remains formal FAIL."
echo "Repairs only the failed expected-vs-actual discrepancy ablation."
echo "A/B labels can route discrepancy but cannot create evidence by themselves."
echo "Zero discrepancy is structurally and bit-exactly [0.5, 0.5]."
echo "Reuses DEV subjects 3000-3255. TEST subjects 6000-6255 remain sealed."

test -s state/autobiographical_self_discrepancy_v1/result.json
test -s state/autobiographical_self_discrepancy_v1/autobiographical_self_model.pt
test -s state/recurrent_self_trajectory_v1/recurrent_self_trajectory.pt
test -s state/recurrent_self_trajectory_test_v1/result.json

python -m py_compile \
  continuous_brain/autobiographical_self_discrepancy_neutral_v1.py \
  train_evaluate_autobiographical_self_discrepancy_neutral_v1.py \
  test_autobiographical_self_discrepancy_neutral_v1.py

python test_autobiographical_self_discrepancy_neutral_v1.py

set +e
python train_evaluate_autobiographical_self_discrepancy_neutral_v1.py \
  --state-dir "$ROOT" \
  2>&1 | tee "$ROOT/run.log"
STATUS=${PIPESTATUS[0]}
set -e

if [ "$STATUS" -ne 0 ] && [ "$STATUS" -ne 4 ]; then
  echo "AUTOBIOGRAPHICAL_SELF_DISCREPANCY_NEUTRAL_WRAPPER: FAIL_EVALUATOR"
  exit "$STATUS"
fi

grep -q '^AUTOBIOGRAPHICAL_SELF_DISCREPANCY_NEUTRAL_COMPLETE$' "$ROOT/run.log"
grep -q '^SOURCE_MODEL_UNCHANGED: true$' "$ROOT/run.log"
grep -q '^PROTECTED_UNCHANGED: true$' "$ROOT/run.log"
grep -q '^AUTOBIO_TEST_TOUCHED: false$' "$ROOT/run.log"

test -s "$ROOT/result.json"
test -s "$ROOT/neutral_autobiographical_self_model.pt"

python - \
  "$ROOT/result.json" \
  "$ROOT/neutral_autobiographical_self_model.pt" \
  "$ROOT/result_bundle.zip" <<'PY'
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import json, sys

r = Path(sys.argv[1])
m = Path(sys.argv[2])
b = Path(sys.argv[3])
p = json.loads(r.read_text())

assert p["source_v0320_formal_status"] == "FAIL"
assert p["source_model_unchanged"] is True
assert p["protected_unchanged"] is True
assert p["autobio_test_touched"] is False
assert p["repair"]["primary_labels_are_routing_only"] is True
assert p["repair"]["all_evidence_layers_bias_free"] is True
assert p["repair"]["zero_discrepancy_maps_exactly_to_half_half"] is True

with ZipFile(b, "w", ZIP_DEFLATED) as z:
    z.write(r, "result.json")
    z.write(m, "neutral_autobiographical_self_model.pt")
    log = r.with_name("run.log")
    if log.is_file():
        z.write(log, "run.log")

print(b)
PY

if [ "$STATUS" -eq 0 ]; then
  echo "AUTOBIOGRAPHICAL_SELF_DISCREPANCY_NEUTRAL_DEV_GATE: PASS_CONFIRMED"
else
  echo "AUTOBIOGRAPHICAL_SELF_DISCREPANCY_NEUTRAL_DEV_GATE: FAIL_CONFIRMED"
fi

echo "AUTOBIOGRAPHICAL_SELF_DISCREPANCY_NEUTRAL_WRAPPER: COMPLETE"
exit "$STATUS"
