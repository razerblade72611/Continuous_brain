#!/usr/bin/env bash
set -euo pipefail
: "${THOR_EXECUTABLE:?Set THOR_EXECUTABLE to the custom AI2-THOR executable}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-$HOME/projects/continuous_brain_v0.3_preserved/state/my_brain/brain.pt}"
STATE_DIR="${STATE_DIR:-state/embodied_brain_v050_clean}"
SCENE="${SCENE:-FloorPlan1Discovery_physics}"
exec continuous-brain \
  --environment thor --scene "$SCENE" --thor-executable "$THOR_EXECUTABLE" \
  --state-dir "$STATE_DIR" --bootstrap-checkpoint "$SOURCE_CHECKPOINT" \
  --curriculum-level 0 --goal-signal teacher-nav --goal-dropout 0.0 \
  --stagnation-steps 60 --vision-backbone "${VISION_BACKBONE:-pixel-stats}" \
  --stop-on-mastery --visual --open-browser "$@"
