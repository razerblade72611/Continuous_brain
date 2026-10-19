#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/continuous_brain_v0.12.2}"
cd "$PROJECT_ROOT"

ROOT="state/continuation_consistent_planning_v1"
mkdir -p "$ROOT"
rm -f "$ROOT/result.json" "$ROOT/run.log" "$ROOT/result_bundle.zip"

echo "=== v0.20.4 Continuation-Consistent Planning Diagnostic ==="
echo "READ ONLY. CPU only. No training."
echo "Reuses spent development seeds 14001-14040."

python evaluate_continuation_consistent_planning_v1.py \
  --output "$ROOT/result.json" \
  2>&1 | tee "$ROOT/run.log"

grep -q '^PROTECTED_UNCHANGED: true$' "$ROOT/run.log"
grep -q '^TRAINING_PERFORMED: false$' "$ROOT/run.log"
grep -q '^V019_REMAINS_CLOSED: true$' "$ROOT/run.log"
grep -q '^SCIENTIFIC_ADMISSION_DECIDED: false$' "$ROOT/run.log"
grep -q '^CONTINUATION_CONSISTENT_PLANNING_RUN: PASS$' "$ROOT/run.log"
test -s "$ROOT/result.json"

python - "$ROOT/result.json" "$ROOT/result_bundle.zip" <<'PY'
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import json,sys
r=Path(sys.argv[1]);b=Path(sys.argv[2]);p=json.loads(r.read_text())
assert p["software_status"]=="PASS"
assert p["training_performed"] is False
assert p["protected_unchanged"] is True
assert p["scientific_admission_decided"] is False
assert p["continuation_consistent"]["illegal_cap_handoffs"] == 0
with ZipFile(b,"w",ZIP_DEFLATED) as z:
    z.write(r,"result.json")
    log=r.with_name("run.log")
    if log.is_file():z.write(log,"run.log")
print(b)
PY

echo "CONTINUATION_CONSISTENT_PLANNING_WRAPPER: PASS"
