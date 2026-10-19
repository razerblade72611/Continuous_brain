#!/usr/bin/env bash
# Run INSIDE WSL. Full suite (~1-2 min on the 5600X). Pass -m "not slow" or a test path to narrow it.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
# shellcheck disable=SC1091
[ -f .venv/bin/activate ] && source .venv/bin/activate
python -m pytest -q "$@"
