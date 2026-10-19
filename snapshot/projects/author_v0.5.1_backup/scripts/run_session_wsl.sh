#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export GALLIUM_DRIVER="${GALLIUM_DRIVER:-d3d12}"
export MESA_D3D12_DEFAULT_ADAPTER_NAME="${MESA_D3D12_DEFAULT_ADAPTER_NAME:-AMD}"

python -m agent_core.cli \
  --scene "${SCENE:-FloorPlan1}" \
  --session --llm --vision --interact \
  --vision-every "${VISION_EVERY:-1}" \
  --step-delay "${STEP_DELAY:-0.08}" \
  --roam-decisions "${DECISIONS:-500}" \
  --max-steps "${MAX_STEPS:-3000}"
