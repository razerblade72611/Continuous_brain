#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source "${AUTHOR_VENV:-$HOME/.venvs/author/bin/activate}"
WINDOWS_HOST="${WINDOWS_HOST:-$(ip route show | awk '/default/ {print $3; exit}')}"
export AGENT_LLM_BASE_URL="${AGENT_LLM_BASE_URL:-http://$WINDOWS_HOST:8080/v1}"
export AGENT_LLM_MODEL="${AGENT_LLM_MODEL:-qwen2-vl-2b}"
THOR_EXE="${THOR_EXE:-/home/davei/projects/ai2thor-humanoid/builds/thor-discovery-Linux64/thor-discovery-Linux64}"
SCENE="${SCENE:-FloorPlan1Discovery_physics}"
GALLIUM_DRIVER="${GALLIUM_DRIVER:-d3d12}" \
MESA_D3D12_DEFAULT_ADAPTER_NAME="${MESA_D3D12_DEFAULT_ADAPTER_NAME:-AMD}" \
python -m agent_core.cli --scene "$SCENE" --session --agents 2 --llm --vision --interact --dashboard \
  --vision-every "${VISION_EVERY:-1}" --step-delay "${STEP_DELAY:-0}" \
  --roam-decisions "${ROAM_DECISIONS:-500}" --max-steps "${MAX_STEPS:-3000}" \
  --peer-text-every "${PEER_TEXT_EVERY:-3}" \
  --thor-executable "$THOR_EXE" "$@"
