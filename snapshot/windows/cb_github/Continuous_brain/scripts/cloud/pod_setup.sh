#!/usr/bin/env bash
# Run ON A RUNPOD POD that David created by hand. Installs the package and runs the tests.
# Launches no further paid resources.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -m pytest -q
