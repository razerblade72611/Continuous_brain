#!/usr/bin/env bash
set -euo pipefail
: "${THOR_EXECUTABLE:?Set THOR_EXECUTABLE to the custom AI2-THOR executable}"
STATE_DIR="${STATE_DIR:-state/embodied_brain_v050_clean}"
SCENE="${SCENE:-FloorPlan1Discovery_physics}"
exec continuous-brain \
  --environment thor --scene "$SCENE" --thor-executable "$THOR_EXECUTABLE" \
  --state-dir "$STATE_DIR" --curriculum-level 1 --goal-signal teacher-nav \
  --goal-dropout 0.0 --stagnation-steps 60 --rehearsal-fraction "${REHEARSAL_FRACTION:-0.20}" \
  --nav-min-path-steps 2 --nav-initial-path-steps 2 --nav-max-path-steps 8 \
  --nav-success-radius 0.12 --nav-alignment-gate-degrees 15 \
  --vision-backbone "${VISION_BACKBONE:-pixel-stats}" --stop-on-mastery \
  --visual --open-browser "$@"
