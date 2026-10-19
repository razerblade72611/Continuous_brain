#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/continuous_brain_v0.12.2}"
cd "$PROJECT_ROOT"

ROOT="state/persistent_self_autobiographical_test_v1"
mkdir -p "$ROOT"
rm -f \
  "$ROOT/result.json" \
  "$ROOT/run.log" \
  "$ROOT/result_bundle.zip" \
  "$ROOT/fresh_test_continuity_probe.pt"

echo "=== v0.28.1 Frozen Persistent-Self TEST ==="
echo "READ ONLY. CPU only. No training."
echo "Uses sealed self-model TEST 30000-33999 exactly once."
echo "Uses fresh causal identity/history seeds not used by v0.28.0 DEV."
echo "After this run self TEST is spent and may not be tuned against."

test -s state/persistent_self_autobiographical_v1/result.json
test -s state/persistent_self_autobiographical_v1/agency_self_model.pt
test -s state/persistent_self_autobiographical_v1/persistent_self_seed.pt

python -m py_compile \
  evaluate_persistent_self_autobiographical_test_v1.py

set +e
python evaluate_persistent_self_autobiographical_test_v1.py \
  --state-dir "$ROOT" \
  2>&1 | tee "$ROOT/run.log"
STATUS=${PIPESTATUS[0]}
set -e

if [ "$STATUS" -ne 0 ] && [ "$STATUS" -ne 4 ]; then
  echo "PERSISTENT_SELF_AUTOBIOGRAPHICAL_TEST_WRAPPER: FAIL_EVALUATOR"
  exit "$STATUS"
fi

grep -q '^PERSISTENT_SELF_AUTOBIOGRAPHICAL_TEST_COMPLETE$' "$ROOT/run.log"
grep -q '^PROTECTED_UNCHANGED: true$' "$ROOT/run.log"
grep -q '^V027_FORMAL_STATUS_PRESERVED: FAIL$' "$ROOT/run.log"
grep -q '^SELF_TEST_SPENT: true$' "$ROOT/run.log"
grep -q '^TEST_TUNING_AFTER_RUN: forbidden$' "$ROOT/run.log"

test -s "$ROOT/result.json"

python - "$ROOT/result.json" "$ROOT/result_bundle.zip" <<'PY'
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import json, sys

r = Path(sys.argv[1])
b = Path(sys.argv[2])
p = json.loads(r.read_text())

assert p["training_performed"] is False
assert p["protected_unchanged"] is True
assert p["v027_formal_status_preserved"] == "FAIL"
assert p["split"]["self_test_spent"] is True
assert p["split"]["test_tuning_after_run_forbidden"] is True
assert p["scope"]["subjective_experience_claimed"] is False
assert p["scope"]["sentience_claimed"] is False
assert p["scope"]["consciousness_claimed"] is False

with ZipFile(b, "w", ZIP_DEFLATED) as z:
    z.write(r, "result.json")
    log = r.with_name("run.log")
    if log.is_file():
        z.write(log, "run.log")

print(b)
PY

if [ "$STATUS" -eq 0 ]; then
  echo "PERSISTENT_SELF_AUTOBIOGRAPHICAL_TEST_ADMISSION: PASS_CONFIRMED"
else
  echo "PERSISTENT_SELF_AUTOBIOGRAPHICAL_TEST_ADMISSION: FAIL_CONFIRMED"
fi

echo "PERSISTENT_SELF_AUTOBIOGRAPHICAL_TEST_WRAPPER: COMPLETE"
exit "$STATUS"
