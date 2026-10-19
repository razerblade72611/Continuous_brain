#!/usr/bin/env bash
# Run INSIDE WSL. Checks the protected base checkpoint against the SHA-256 recorded in the dossier.
set -euo pipefail
ROOT="${1:-$HOME/projects/continuous_brain_v0.12.2}"
EXPECTED=086932388a9a410af46fea8e87cc63dd4b60161e03e247bf81fc88d28cfd577f
FILE="$ROOT/state/child_v113_dml_smoke/infant_training.pt"
ACTUAL="$(sha256sum "$FILE" | cut -d' ' -f1)"
echo "expected $EXPECTED"
echo "actual   $ACTUAL"
if [ "$ACTUAL" = "$EXPECTED" ]; then echo "MATCH"; else echo "MISMATCH"; exit 1; fi
