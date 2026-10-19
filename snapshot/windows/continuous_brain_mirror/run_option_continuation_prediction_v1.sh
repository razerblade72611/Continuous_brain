#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/continuous_brain_v0.12.2}"
cd "$PROJECT_ROOT"
export PYTHONDONTWRITEBYTECODE=1

ROOT="state/option_continuation_prediction_v1"
mkdir -p "$ROOT"
RUN_DIR="$(mktemp -d "$ROOT/run_XXXXXXXX")"

echo "=== Option continuation and prediction diagnostic ==="
echo "Read-only checkpoints. CPU. Same 80 development boundaries."
echo "Run directory: $RUN_DIR"

if ! python -u diagnose_option_continuation_prediction_v1.py \
    --project-root "$PROJECT_ROOT" --output "$RUN_DIR/report.json" \
    2>&1 | tee "$RUN_DIR/run.log"; then
    echo "OPTION_CONTINUATION_WRAPPER: ERROR"
    echo "Details: $RUN_DIR/run.log"
    exit 1
fi

if ! python diagnose_option_continuation_prediction_v1.py \
    --verify-report "$RUN_DIR/report.json"; then
    echo "OPTION_CONTINUATION_WRAPPER: ERROR"
    exit 1
fi

python - "$RUN_DIR" <<'PY'
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import sys
root=Path(sys.argv[1])
with ZipFile(root/'result_bundle.zip','x',ZIP_DEFLATED) as archive:
    for name in ['report.json','boundaries.jsonl','run.log']:
        archive.write(root/name,name)
print('Upload:',root/'result_bundle.zip')
PY

echo "OPTION_CONTINUATION_WRAPPER: COMPLETE"
