#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/continuous_brain_v0.12.2}"
cd "$PROJECT_ROOT"

ROOT="state/self_broadcast_centrality_test_v1"
mkdir -p "$ROOT"
rm -f \
  "$ROOT/result.json" \
  "$ROOT/run.log" \
  "$ROOT/result_bundle.zip"

echo "=== v0.29.3 Frozen Self-Broadcast Centrality TEST ==="
echo "READ ONLY. CPU only. No training."
echo "Uses frozen v0.29.0 shared hub + v0.29.2 gradient-safe priority head."
echo "Uses sealed broadcast TEST 30000-32999 exactly once."
echo "Uses fresh donor/history/episodic-routing interventions."
echo "After this run broadcast TEST is spent and may not be tuned against."

test -s state/self_broadcast_centrality_v1/result.json
test -s state/self_broadcast_centrality_v1/self_broadcast_hub.pt
test -s state/self_broadcast_value_priority_gradient_safe_v1/result.json
test -s state/self_broadcast_value_priority_gradient_safe_v1/value_priority_gradient_safe.pt
test -s state/persistent_self_autobiographical_test_v1/result.json

python -m py_compile \
  evaluate_self_broadcast_centrality_test_v1.py

set +e
python evaluate_self_broadcast_centrality_test_v1.py \
  --state-dir "$ROOT" \
  2>&1 | tee "$ROOT/run.log"
STATUS=${PIPESTATUS[0]}
set -e

if [ "$STATUS" -ne 0 ] && [ "$STATUS" -ne 4 ]; then
  echo "SELF_BROADCAST_CENTRALITY_TEST_WRAPPER: FAIL_EVALUATOR"
  exit "$STATUS"
fi

grep -q '^SELF_BROADCAST_CENTRALITY_TEST_COMPLETE$' "$ROOT/run.log"
grep -q '^PROTECTED_UNCHANGED: true$' "$ROOT/run.log"
grep -q '^SOURCE_HUB_UNCHANGED: true$' "$ROOT/run.log"
grep -q '^PRIORITY_HEAD_UNCHANGED: true$' "$ROOT/run.log"
grep -q '^BROADCAST_TEST_SPENT: true$' "$ROOT/run.log"
grep -q '^TEST_TUNING_AFTER_RUN: forbidden$' "$ROOT/run.log"

test -s "$ROOT/result.json"

python - \
  "$ROOT/result.json" \
  "$ROOT/result_bundle.zip" <<'PY'
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import json, sys

r = Path(sys.argv[1])
b = Path(sys.argv[2])
p = json.loads(r.read_text())

assert p["training_performed"] is False
assert p["protected_unchanged"] is True
assert p["source_hub_unchanged"] is True
assert p["priority_head_unchanged"] is True
assert p["historical_status"]["v0290"] == "FAIL"
assert p["historical_status"]["v0291"] == "FAIL"
assert p["historical_status"]["v0292_dev_repair"] == "PASS"
assert p["split"]["test_spent"] is True
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
  echo "SELF_BROADCAST_CENTRALITY_TEST_ADMISSION: PASS_CONFIRMED"
else
  echo "SELF_BROADCAST_CENTRALITY_TEST_ADMISSION: FAIL_CONFIRMED"
fi

echo "SELF_BROADCAST_CENTRALITY_TEST_WRAPPER: COMPLETE"
exit "$STATUS"
