#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/continuous_brain_v0.12.2}"
cd "$PROJECT_ROOT"

ROOT="state/option_support_counterfactual_v1"
mkdir -p "$ROOT"

rm -f "$ROOT/result.json" "$ROOT/run.log" "$ROOT/result_bundle.zip"

echo "=== v0.20.3 Option Support Counterfactual Diagnostic ==="
echo "READ ONLY. CPU only. No training."
echo "Reuses spent v0.20.2 development window."

python diagnose_option_support_counterfactual_v1.py \
  --output "$ROOT/result.json" \
  2>&1 | tee "$ROOT/run.log"

grep -q '^PROTECTED_UNCHANGED: true$' "$ROOT/run.log"
grep -q '^TRAINING_PERFORMED: false$' "$ROOT/run.log"
grep -q '^V019_REMAINS_CLOSED: true$' "$ROOT/run.log"
grep -q '^OPTION_SUPPORT_COUNTERFACTUAL_RUN: PASS$' "$ROOT/run.log"
test -s "$ROOT/result.json"

python - "$ROOT/result.json" "$ROOT/result_bundle.zip" <<'PY'
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import json, sys

r=Path(sys.argv[1]); b=Path(sys.argv[2])
p=json.loads(r.read_text())
assert p["software_status"]=="PASS"
assert p["training_performed"] is False
assert p["protected_unchanged"] is True
assert p["v019_remains_closed"] is True

with ZipFile(b,"w",ZIP_DEFLATED) as z:
    z.write(r,"result.json")
    log=r.with_name("run.log")
    if log.is_file():
        z.write(log,"run.log")
print(b)
PY

echo "OPTION_SUPPORT_COUNTERFACTUAL_WRAPPER: PASS"
