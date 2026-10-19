#!/usr/bin/env bash
# Run INSIDE WSL. Read-only DAV-74 inventory of the CANONICAL legacy tree (not the Windows mirror).
# Writes legacy/candidates_wsl.json in this repo; nothing is written under the legacy tree.
#   bash scripts/local/inventory_legacy.sh                       # default root below
#   bash scripts/local/inventory_legacy.sh /path/to/legacy/tree
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROOT="${1:-$HOME/projects/continuous_brain_v0.12.2}"
python3 "$REPO/continuous_brain_unified/legacy/inventory.py" \
  --root "$ROOT" --spec "$REPO/legacy/candidate_spec.json" \
  --root-label wsl-canonical --out "$REPO/legacy/candidates_wsl.json"
sha256sum "$REPO/legacy/candidates_wsl.json"
