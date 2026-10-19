#!/usr/bin/env bash
# Read-only refresh of DAV-74 candidate parsing against the canonical WSL tree.
# Output stays in this repository; nothing is written below the legacy root.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROOT="${1:-${CONTINUOUS_BRAIN_LEGACY_ROOT:-$HOME/projects/continuous_brain_v0.12.2}}"
OUTDIR="$REPO/evidence/g0_runtime"
mkdir -p "$OUTDIR"
OUT="$OUTDIR/candidates_wsl_v0381.json"
python3 "$REPO/continuous_brain_unified/legacy/inventory.py" \
  --root "$ROOT" --spec "$REPO/legacy/candidate_spec.json" \
  --root-label wsl-canonical --out "$OUT"
sha256sum "$OUT" | tee "$OUT.sha256"
python3 "$REPO/scripts/local/g0_plan.py" --legacy-root "$ROOT"
echo "refresh written: $OUT"
