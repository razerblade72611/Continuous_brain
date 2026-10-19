#!/usr/bin/env bash
set -euo pipefail
: "${THOR_EXECUTABLE:?Set THOR_EXECUTABLE to your local AI2-THOR executable}"
SCENE="${SCENE:-FloorPlan1Discovery_physics}"
LEVEL="${LEVEL:-1}"
STATE_DIR="${STATE_DIR:-state/embodied_brain_v050_spatial}"
exec continuous-brain \
  --environment thor \
  --scene "$SCENE" \
  --thor-executable "$THOR_EXECUTABLE" \
  --curriculum-level "$LEVEL" \
  --state-dir "$STATE_DIR" \
  --vision-backbone "${VISION_BACKBONE:-pixel-stats}" \
  --vision-every "${VISION_EVERY:-1}" \
  --rehearsal-fraction "${REHEARSAL_FRACTION:-0.15}" \
  --visual --open-browser "$@"
