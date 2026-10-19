#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/continuous_brain_v0.12.2}"
cd "$PROJECT_ROOT"

ROOT="state/self_broadcast_centrality_v1"
mkdir -p "$ROOT"
rm -f   "$ROOT/result.json"   "$ROOT/run.log"   "$ROOT/result_bundle.zip"   "$ROOT/self_broadcast_hub.pt"

echo "=== v0.29.0 Persistent Self Broadcast / Causal Centrality ==="
echo "Cognition milestone. No robot training."
echo "Trains only one additive shared self-broadcast hub."
echo "Frozen v0.28 persistent self feeds workspace, retrieval, goal and planning consumers."
echo "Broadcast TEST 30000-32999 remains sealed."

test -s state/persistent_self_autobiographical_v1/agency_self_model.pt
test -s state/persistent_self_autobiographical_v1/persistent_self_seed.pt
test -s state/persistent_self_autobiographical_test_v1/result.json
test -s state/multi_item_episodic_memory_v1/episodic_reader.pt
test -s state/child_v113_dml_smoke/infant_training.pt

python -m py_compile   continuous_brain/self_broadcast_centrality_v1.py   train_evaluate_self_broadcast_centrality_v1.py   test_self_broadcast_centrality_v1.py

python test_self_broadcast_centrality_v1.py

set +e
python train_evaluate_self_broadcast_centrality_v1.py   --state-dir "$ROOT"   2>&1 | tee "$ROOT/run.log"
STATUS=${PIPESTATUS[0]}
set -e

if [ "$STATUS" -ne 0 ] && [ "$STATUS" -ne 4 ]; then
  echo "SELF_BROADCAST_CENTRALITY_WRAPPER: FAIL_EVALUATOR"
  exit "$STATUS"
fi

grep -q '^SELF_BROADCAST_CENTRALITY_RUN_COMPLETE$' "$ROOT/run.log"
grep -q '^PROTECTED_UNCHANGED: true$' "$ROOT/run.log"
grep -q '^V0281_SELF_TEST_PRESERVED: true$' "$ROOT/run.log"
grep -q '^BROADCAST_TEST_TOUCHED: false$' "$ROOT/run.log"

test -s "$ROOT/result.json"
test -s "$ROOT/self_broadcast_hub.pt"

python -   "$ROOT/result.json"   "$ROOT/self_broadcast_hub.pt"   "$ROOT/result_bundle.zip" <<'PY'
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import json, sys

r = Path(sys.argv[1])
h = Path(sys.argv[2])
b = Path(sys.argv[3])

p = json.loads(r.read_text())
assert p["protected_unchanged"] is True
assert p["v0281_self_test_preserved"] is True
assert p["split"]["test_touched"] is False
assert p["training"]["self_broadcast_hub"] is True
assert p["training"]["historical_brain"] is False
assert p["training"]["persistent_self"] is False
assert p["architecture"]["broadcast_off_exact_neutral"] is True

with ZipFile(b, "w", ZIP_DEFLATED) as z:
    z.write(r, "result.json")
    z.write(h, "self_broadcast_hub.pt")
    log = r.with_name("run.log")
    if log.is_file():
        z.write(log, "run.log")

print(b)
PY

if [ "$STATUS" -eq 0 ]; then
  echo "SELF_BROADCAST_CENTRALITY_DEV_GATE: PASS_CONFIRMED"
else
  echo "SELF_BROADCAST_CENTRALITY_DEV_GATE: FAIL_CONFIRMED"
fi

echo "SELF_BROADCAST_CENTRALITY_WRAPPER: COMPLETE"
exit "$STATUS"
