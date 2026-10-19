#!/usr/bin/env bash
set -euo pipefail
cat <<'EOF'
This script DOES NOT launch a RunPod instance.
Suggested manual sequence after you choose/approve a pod:
  git clone <your-repo> continuous_brain_v038
  cd continuous_brain_v038
  python -m venv .venv && source .venv/bin/activate
  pip install -e '.[dev]'
  pytest -q
Record GPU model, driver/runtime, commit, config hash, seeds, wall time and cost in the run manifest.
EOF
