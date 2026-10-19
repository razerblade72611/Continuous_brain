#!/usr/bin/env bash
set -euo pipefail
: "${THOR_EXECUTABLE:?Set THOR_EXECUTABLE to your local AI2-THOR executable}"
SCENE="${SCENE:-FloorPlan1Discovery_physics}"
STATE_DIR="${STATE_DIR:-state/embodied_brain}"
LEVEL="${LEVEL:-0}"
VISION_BACKBONE="${VISION_BACKBONE:-pixel-stats}"
GOAL_DROPOUT="${GOAL_DROPOUT:-0.0}"
STAGNATION_STEPS="${STAGNATION_STEPS:-60}"

echo "=== Continuous Brain v0.4.1 curriculum L${LEVEL} ==="
echo "v0.4.1 does not blindly advance levels. Graduate only after the live/session"
echo "window meets the criterion (default: >=95% success, median <=6 steps, <=2% timeouts)."
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
  --visual --open-browser "$@"
