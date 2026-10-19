#!/usr/bin/env bash
set -euo pipefail
WINDOWS_HOST="${WINDOWS_HOST:-$(ip route show | awk '/default/ {print $3; exit}')}"
export AGENT_LLM_BASE_URL="${AGENT_LLM_BASE_URL:-http://${WINDOWS_HOST}:8080/v1}"
export AGENT_LLM_MODEL="${AGENT_LLM_MODEL:-qwen2-vl-2b}"
export GALLIUM_DRIVER="${GALLIUM_DRIVER:-d3d12}"
export MESA_D3D12_DEFAULT_ADAPTER_NAME="${MESA_D3D12_DEFAULT_ADAPTER_NAME:-AMD}"
python -m agent_core.cli --scene FloorPlan1 --session --agents 2 --llm --vision --interact "$@"
