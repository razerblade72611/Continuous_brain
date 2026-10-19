#!/usr/bin/env bash
# Run INSIDE WSL (Ubuntu), from anywhere. Creates .venv in the repo, installs the package
# (editable, with dev extras), and runs the test suite. CPU-only torch; add DirectML separately.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest -q
echo "ok: activate later with  source $REPO/.venv/bin/activate"
