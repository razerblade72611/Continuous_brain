#!/usr/bin/env bash
set -euo pipefail
: "${THOR_EXECUTABLE:?Set THOR_EXECUTABLE to the custom AI2-THOR executable}"
STATE_DIR="${STATE_DIR:-state/embodied_brain_v043_clean}"
SCENE="${SCENE:-FloorPlan1Discovery_physics}"
exec continuous-brain \
  --environment thor \
  --scene "$SCENE" \
  --thor-executable "$THOR_EXECUTABLE" \
  --state-dir "$STATE_DIR" \
  --curriculum-level 1 \
  --goal-signal teacher-nav \
  --goal-dropout 0.0 \
  --stagnation-steps 60 \
  --rehearsal-fraction "${REHEARSAL_FRACTION:-0.20}" \
  --nav-min-path-steps "${NAV_MIN_PATH_STEPS:-2}" \
  --nav-initial-path-steps "${NAV_INITIAL_PATH_STEPS:-2}" \
  --nav-max-path-steps "${NAV_MAX_PATH_STEPS:-8}" \
  --nav-success-radius "${NAV_SUCCESS_RADIUS:-0.12}" \
  --nav-alignment-gate-degrees "${NAV_ALIGNMENT_GATE_DEGREES:-15}" \
  --vision-backbone "${VISION_BACKBONE:-pixel-stats}" \
  --stop-on-mastery \
  --visual --open-browser "$@"
