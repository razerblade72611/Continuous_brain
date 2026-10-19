#!/usr/bin/env bash
set -euo pipefail
cd "${PROJECT_ROOT:-$HOME/projects/continuous_brain_v0.12.2}"
ROOT="state/temporal_option_affordance_v1"
mkdir -p "$ROOT"
rm -f "$ROOT/result.json" "$ROOT/run.log" "$ROOT/result_bundle.zip"

echo "=== v0.21.1 Temporal Option Affordance Diagnostic ==="
echo "READ ONLY. CPU only. No training."
echo "Reuses spent v0.21 DEV conditions 1600-1727."
echo "Enumerates all learned options at identical delayed-decision states."

python -m py_compile diagnose_temporal_option_affordance_v1.py
python diagnose_temporal_option_affordance_v1.py --output "$ROOT/result.json" 2>&1 | tee "$ROOT/run.log"

grep -q '^TEMPORAL_OPTION_AFFORDANCE_COMPLETE$' "$ROOT/run.log"
grep -q '^PROTECTED_UNCHANGED: true$' "$ROOT/run.log"
grep -q '^TRAINING_PERFORMED: false$' "$ROOT/run.log"
grep -q '^TEST_TOUCHED: false$' "$ROOT/run.log"

python - "$ROOT/result.json" "$ROOT/result_bundle.zip" <<'PY'
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import json, sys
r=Path(sys.argv[1]); b=Path(sys.argv[2])
p=json.loads(r.read_text())
assert p["training_performed"] is False
assert p["new_development_window_consumed"] is False
assert p["test_touched"] is False
assert p["protected_unchanged"] is True
with ZipFile(b,"w",ZIP_DEFLATED) as z:
    z.write(r,"result.json")
    log=r.with_name("run.log")
    if log.is_file(): z.write(log,"run.log")
print(b)
PY
echo "TEMPORAL_OPTION_AFFORDANCE_WRAPPER: COMPLETE"
