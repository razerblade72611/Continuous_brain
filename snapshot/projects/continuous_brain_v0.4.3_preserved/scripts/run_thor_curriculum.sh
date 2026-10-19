#!/usr/bin/env bash
set -euo pipefail
: "${THOR_EXECUTABLE:?Set THOR_EXECUTABLE to your local AI2-THOR executable}"
SCENE="${SCENE:-FloorPlan1Discovery_physics}"
STATE_DIR="${STATE_DIR:-state/embodied_brain_v043_clean}"
LEVEL="${LEVEL:-1}"
VISION_BACKBONE="${VISION_BACKBONE:-pixel-stats}"
GOAL_DROPOUT="${GOAL_DROPOUT:-0.0}"
STAGNATION_STEPS="${STAGNATION_STEPS:-60}"
REHEARSAL_FRACTION="${REHEARSAL_FRACTION:-0.15}"
NAV_MIN_PATH_STEPS="${NAV_MIN_PATH_STEPS:-2}"
NAV_INITIAL_PATH_STEPS="${NAV_INITIAL_PATH_STEPS:-2}"
NAV_MAX_PATH_STEPS="${NAV_MAX_PATH_STEPS:-8}"
NAV_SUCCESS_RADIUS="${NAV_SUCCESS_RADIUS:-0.12}"
NAV_ALIGNMENT_GATE_DEGREES="${NAV_ALIGNMENT_GATE_DEGREES:-15}"

echo "=== Continuous Brain v0.4.3 curriculum L${LEVEL} ==="
echo "Current-level graduation uses level-local evidence only. L1 uses reachable-grid"
echo "waypoints, real-motion success, alignment-gated advance, native collision evidence, persistent replay and skill snapshots."
exec continuous-brain \
  --environment thor \
  --scene "$SCENE" \
  --thor-executable "$THOR_EXECUTABLE" \
  --curriculum-level "$LEVEL" \
  --state-dir "$STATE_DIR" \
  --vision-backbone "$VISION_BACKBONE" \
  --goal-signal teacher-nav \
  --goal-dropout "$GOAL_DROPOUT" \
  --stagnation-steps "$STAGNATION_STEPS" \
  --rehearsal-fraction "$REHEARSAL_FRACTION" \
  --nav-min-path-steps "$NAV_MIN_PATH_STEPS" \
  --nav-initial-path-steps "$NAV_INITIAL_PATH_STEPS" \
  --nav-max-path-steps "$NAV_MAX_PATH_STEPS" \
  --nav-success-radius "$NAV_SUCCESS_RADIUS" \
  --nav-alignment-gate-degrees "$NAV_ALIGNMENT_GATE_DEGREES" \
  --visual --open-browser "$@"
