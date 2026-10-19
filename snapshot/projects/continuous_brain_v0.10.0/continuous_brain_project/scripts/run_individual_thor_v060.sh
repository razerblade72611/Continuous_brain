#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${THOR_EXECUTABLE:?Set THOR_EXECUTABLE to the custom AI2-THOR executable}"
exec python -m continuous_brain.individual_cli \
  --environment thor --profile "${PROFILE:-standard}" \
  --state-dir "${STATE_DIR:-state/individual_thor_v060}" \
  --scene "${SCENE:-FloorPlan1Discovery_physics}" \
  --thor-executable "$THOR_EXECUTABLE" --curriculum-level 2 --goal-signal internal-map \
  --vision-backbone "${VISION_BACKBONE:-pixel-stats}" \
  --visual --open-browser "$@"
