#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

THOR_EXE_DEFAULT="/home/davei/projects/ai2thor-humanoid/builds/thor-humanoid-Linux64/thor-humanoid-Linux64"
THOR_EXE="${THOR_EXE:-$THOR_EXE_DEFAULT}"
EPISODES="${EPISODES:-500}"
WORKERS="${WORKERS:-1}"
STATE_DIR="${STATE_DIR:-state}"

# Set RESET_MEMORY=1 for a clean training lineage. This deletes only STATE_DIR.
RESET_ARGS=()
if [[ "${RESET_MEMORY:-0}" == "1" ]]; then
  RESET_ARGS+=(--reset-memory)
fi

GALLIUM_DRIVER="${GALLIUM_DRIVER:-d3d12}" \
MESA_D3D12_DEFAULT_ADAPTER_NAME="${MESA_D3D12_DEFAULT_ADAPTER_NAME:-AMD}" \
python -m agent_core.training \
  --config config.yaml \
  --scene FloorPlan1 \
  --episodes "$EPISODES" \
  --workers "$WORKERS" \
  --state-dir "$STATE_DIR" \
  --thor-executable "$THOR_EXE" \
  "${RESET_ARGS[@]}" \
  "$@"
