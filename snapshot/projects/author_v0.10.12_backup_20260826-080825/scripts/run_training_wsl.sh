#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

THOR_EXE_DEFAULT="/home/davei/projects/ai2thor-humanoid/builds/thor-discovery-Linux64/thor-discovery-Linux64"
THOR_EXE="${THOR_EXE:-$THOR_EXE_DEFAULT}"
EPISODES="${EPISODES:-1000}"
WORKERS="${WORKERS:-1}"
STATE_DIR="${STATE_DIR:-state}"
SCENE="${SCENE:-FloorPlan1Discovery_physics}"

EXTRA_ARGS=()
if [[ "${RESET_MEMORY:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--reset-memory)
fi
if [[ -n "${CURRICULUM:-}" ]]; then
  EXTRA_ARGS+=(--curriculum "$CURRICULUM")
fi
if [[ -n "${LEVEL:-}" ]]; then
  EXTRA_ARGS+=(--level "$LEVEL")
fi
if [[ "${NO_SFT_EXPORT:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--no-sft-export)
fi

GALLIUM_DRIVER="${GALLIUM_DRIVER:-d3d12}" \
MESA_D3D12_DEFAULT_ADAPTER_NAME="${MESA_D3D12_DEFAULT_ADAPTER_NAME:-AMD}" \
python -m agent_core.training \
  --config config.yaml \
  --scene "$SCENE" \
  --episodes "$EPISODES" \
  --workers "$WORKERS" \
  --state-dir "$STATE_DIR" \
  --thor-executable "$THOR_EXE" \
  "${EXTRA_ARGS[@]}" \
  "$@"
