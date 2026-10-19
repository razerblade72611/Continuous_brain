#!/usr/bin/env bash
set -euo pipefail
: "${THOR_EXECUTABLE:?Set THOR_EXECUTABLE to the custom AI2-THOR executable}"
STATE_DIR="${STATE_DIR:-state/embodied_brain_v050_spatial}"
SCENE="${SCENE:-FloorPlan1Discovery_physics}"
exec continuous-brain \
  --environment thor \
  --scene "$SCENE" \
  --thor-executable "$THOR_EXECUTABLE" \
  --state-dir "$STATE_DIR" \
  --curriculum-level 2 \
  --goal-signal internal-map \
  --goal-dropout 0.0 \
  --stagnation-steps "${STAGNATION_STEPS:-60}" \
  --rehearsal-fraction "${REHEARSAL_FRACTION:-0.20}" \
  --nav-alignment-gate-degrees "${NAV_ALIGNMENT_GATE_DEGREES:-15}" \
  --spatial-resolution "${SPATIAL_RESOLUTION:-0.25}" \
  --spatial-depth-min-trusted "${SPATIAL_DEPTH_MIN_TRUSTED:-0.25}" \
  --spatial-depth-rays "${SPATIAL_DEPTH_RAYS:-31}" \
  --spatial-waypoint-lookahead-cells "${SPATIAL_WAYPOINT_LOOKAHEAD_CELLS:-2}" \
  --spatial-replan-every-steps "${SPATIAL_REPLAN_EVERY_STEPS:-4}" \
  --spatial-target-min-distance "${SPATIAL_TARGET_MIN_DISTANCE:-1.0}" \
  --spatial-target-max-distance "${SPATIAL_TARGET_MAX_DISTANCE:-2.5}" \
  --spatial-min-successful-translations "${SPATIAL_MIN_SUCCESSFUL_TRANSLATIONS:-4}" \
  --spatial-success-radius "${SPATIAL_SUCCESS_RADIUS:-0.18}" \
  --vision-backbone "${VISION_BACKBONE:-pixel-stats}" \
  --stop-on-mastery \
  --visual --open-browser "$@"
