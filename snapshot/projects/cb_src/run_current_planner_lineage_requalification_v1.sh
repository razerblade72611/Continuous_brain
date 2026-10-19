#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/continuous_brain_v0.12.2}"
cd "$PROJECT_ROOT"

ROOT="state/planner_lineage_requalification_v1"
mkdir -p "$ROOT"

rm -f \
  "$ROOT/current_smoke_9001_9010.json" \
  "$ROOT/current_final_12001_12100.json" \
  "$ROOT/result.json" \
  "$ROOT/run.log" \
  "$ROOT/audit_bundle.zip"

DEVICE="${CB_PLANNER_REQUAL_DEVICE:-auto}"

echo "=== v0.19.2 Current Planner Lineage Requalification ==="
echo "READ ONLY: no training, no model checkpoint writes."
echo "device=$DEVICE"

set +e
python requalify_current_planner_lineage_v1.py \
  --output-dir "$ROOT" \
  --device "$DEVICE" \
  2>&1 | tee "$ROOT/run.log"
STATUS=${PIPESTATUS[0]}
set -e

# Never accept a stale/missing result.
if [ "$STATUS" -ne 0 ]; then
  echo "PLANNER_LINEAGE_REQUALIFICATION_RUN: FAIL"
  exit "$STATUS"
fi

grep -q '^CURRENT_PLANNER_P4B_REQUALIFICATION: PASS$' "$ROOT/run.log"
grep -q '^PROTECTED_UNCHANGED: True$' "$ROOT/run.log"
grep -q '^V019_REMAINS_CLOSED: true$' "$ROOT/run.log"
grep -q '^TRAINING_PERFORMED: false$' "$ROOT/run.log"
grep -q '^GENERALIZATION_ESTABLISHED: false$' "$ROOT/run.log"

test -s "$ROOT/result.json"
test -s "$ROOT/current_smoke_9001_9010.json"
test -s "$ROOT/current_final_12001_12100.json"

python - "$ROOT/result.json" "$ROOT/audit_bundle.zip" <<'PY'
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import json, sys

result = Path(sys.argv[1])
bundle = Path(sys.argv[2])
data = json.loads(result.read_text())

assert data["status"] == "PASS"
assert data["read_only"] is True
assert data["training_performed"] is False
assert data["protected_unchanged"] is True
assert data["v019_remains_closed"] is True
assert data["admitted_memory_preserved"] is True
assert data["v018_intent_preserved"] is True
assert data["generalization_established"] is False
assert data["current_final"]["p4b_passed"] is True

root = result.parent
with ZipFile(bundle, "w", ZIP_DEFLATED) as z:
    for name in (
        "result.json",
        "current_smoke_9001_9010.json",
        "current_final_12001_12100.json",
        "run.log",
    ):
        p = root / name
        if p.is_file():
            z.write(p, name)
print(bundle)
PY

echo "PLANNER_LINEAGE_REQUALIFICATION_RUN: PASS"
