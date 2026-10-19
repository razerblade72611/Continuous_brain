#!/usr/bin/env bash
set -euo pipefail

SCENE="${1:-FloorPlan1}"
DECISIONS="${2:-100}"
MAX_STEPS="${3:-500}"

export GALLIUM_DRIVER="${GALLIUM_DRIVER:-d3d12}"
export MESA_D3D12_DEFAULT_ADAPTER_NAME="${MESA_D3D12_DEFAULT_ADAPTER_NAME:-AMD}"

python -m agent_core.cli \
  --scene "$SCENE" \
  --roam \
  --llm \
  --vision \
  --roam-decisions "$DECISIONS" \
  --max-steps "$MAX_STEPS"
