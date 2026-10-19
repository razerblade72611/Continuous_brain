#!/usr/bin/env bash
set -euo pipefail

GOAL=${1:-"Pick up the Apple and put it on the CounterTop."}
SCENE=${2:-"FloorPlan1"}

export GALLIUM_DRIVER=${GALLIUM_DRIVER:-d3d12}
export MESA_D3D12_DEFAULT_ADAPTER_NAME=${MESA_D3D12_DEFAULT_ADAPTER_NAME:-AMD}

python -m agent_core.cli --scene "$SCENE" --goal "$GOAL"
