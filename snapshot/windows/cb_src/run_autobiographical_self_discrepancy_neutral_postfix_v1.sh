#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/continuous_brain_v0.12.2}"
cd "$PROJECT_ROOT"

ROOT="state/autobiographical_self_discrepancy_neutral_v1"
mkdir -p "$ROOT"
rm -f \
  "$ROOT/result.json" \
  "$ROOT/postfix_run.log" \
  "$ROOT/postfix_result_bundle.zip"

echo "=== v0.32.1a Autobiographical Neutral Evaluator Postfix ==="
echo "INFRASTRUCTURE REPAIR ONLY. NO TRAINING."
echo "Reuses the already-trained v0.32.1 neutral autobiographical model."
echo "Fixes only the accidental torch.no_grad() around the gradient sanity check."
echo "Autobiographical TEST subjects 6000-6255 remain sealed."

test -s "$ROOT/neutral_autobiographical_self_model.pt"
test -s state/autobiographical_self_discrepancy_v1/result.json
test -s state/recurrent_self_trajectory_v1/recurrent_self_trajectory.pt
test -s state/recurrent_self_trajectory_test_v1/result.json

python -m py_compile \
  evaluate_autobiographical_self_discrepancy_neutral_postfix_v1.py

set +e
python evaluate_autobiographical_self_discrepancy_neutral_postfix_v1.py \
  --state-dir "$ROOT" \
  2>&1 | tee "$ROOT/postfix_run.log"
STATUS=${PIPESTATUS[0]}
set -e

if [ "$STATUS" -ne 0 ] && [ "$STATUS" -ne 4 ]; then
  echo "AUTOBIOGRAPHICAL_SELF_DISCREPANCY_NEUTRAL_POSTFIX_WRAPPER: FAIL_EVALUATOR"
  exit "$STATUS"
fi

grep -q '^AUTOBIOGRAPHICAL_SELF_DISCREPANCY_NEUTRAL_POSTFIX_COMPLETE$' \
  "$ROOT/postfix_run.log"
grep -q '^TRAINING_PERFORMED: false$' "$ROOT/postfix_run.log"
grep -q '^NEUTRAL_MODEL_UNCHANGED: true$' "$ROOT/postfix_run.log"
grep -q '^PROTECTED_UNCHANGED: true$' "$ROOT/postfix_run.log"
grep -q '^AUTOBIO_TEST_TOUCHED: false$' "$ROOT/postfix_run.log"

test -s "$ROOT/result.json"

python - \
  "$ROOT/result.json" \
  "$ROOT/neutral_autobiographical_self_model.pt" \
  "$ROOT/postfix_result_bundle.zip" <<'PY'
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import json, sys

r = Path(sys.argv[1])
m = Path(sys.argv[2])
b = Path(sys.argv[3])
p = json.loads(r.read_text())

assert p["training_performed"] is False
assert p["infrastructure_repair"]["model_retrained"] is False
assert p["neutral_model_unchanged"] is True
assert p["protected_unchanged"] is True
assert p["autobio_test_touched"] is False

with ZipFile(b, "w", ZIP_DEFLATED) as z:
    z.write(r, "result.json")
    z.write(m, "neutral_autobiographical_self_model.pt")
    log = r.with_name("postfix_run.log")
    if log.is_file():
        z.write(log, "postfix_run.log")

print(b)
PY

if [ "$STATUS" -eq 0 ]; then
  echo "AUTOBIOGRAPHICAL_SELF_DISCREPANCY_NEUTRAL_DEV_GATE: PASS_CONFIRMED"
else
  echo "AUTOBIOGRAPHICAL_SELF_DISCREPANCY_NEUTRAL_DEV_GATE: FAIL_CONFIRMED"
fi

echo "AUTOBIOGRAPHICAL_SELF_DISCREPANCY_NEUTRAL_POSTFIX_WRAPPER: COMPLETE"
exit "$STATUS"
