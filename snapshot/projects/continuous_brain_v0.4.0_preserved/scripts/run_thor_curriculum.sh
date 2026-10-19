#!/usr/bin/env bash
set -euo pipefail
: "${THOR_EXECUTABLE:?Set THOR_EXECUTABLE to your local AI2-THOR executable}"
SCENE="${SCENE:-FloorPlan1Discovery_physics}"
STATE_DIR="${STATE_DIR:-state/embodied_brain}"
STEPS_PER_LEVEL="${STEPS_PER_LEVEL:-5000}"
VISION_BACKBONE="${VISION_BACKBONE:-pixel-stats}"
GOAL_DROPOUT="${GOAL_DROPOUT:-0.15}"
for LEVEL in 0 1 2 3 4 5 6 7 8 9; do
  echo "=== Continuous Brain embodied curriculum L${LEVEL} ==="
  continuous-brain \
    --environment thor \
    --scene "$SCENE" \
    --thor-executable "$THOR_EXECUTABLE" \
    --curriculum-level "$LEVEL" \
    --state-dir "$STATE_DIR" \
    --vision-backbone "$VISION_BACKBONE" \
    --goal-signal teacher-nav \
    --goal-dropout "$GOAL_DROPOUT" \
    --additional-steps "$STEPS_PER_LEVEL" \
    --visual "$@"
done
