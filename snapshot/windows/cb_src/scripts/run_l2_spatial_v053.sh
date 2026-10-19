#!/usr/bin/env bash
set -euo pipefail
: "${THOR_EXECUTABLE:?Set THOR_EXECUTABLE to the custom AI2-THOR executable}"
STATE_DIR="${STATE_DIR:-state/embodied_brain_v053_spatial}"
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
  --sensor-depth-min-trusted "${SENSOR_DEPTH_MIN_TRUSTED:-0.25}" \
  --sensor-rgb-diff-threshold "${SENSOR_RGB_DIFF_THRESHOLD:-0.25}" \
  --sensor-rgb-stale-motion-events "${SENSOR_RGB_STALE_MOTION_EVENTS:-3}" \
  --sensor-depth-unhealthy-events "${SENSOR_DEPTH_UNHEALTHY_EVENTS:-3}" \
  --sensor-recovery-cooldown-ticks "${SENSOR_RECOVERY_COOLDOWN_TICKS:-120}" \
  --sensor-soft-reset-window-seconds "${SENSOR_SOFT_RESET_WINDOW_SECONDS:-600}" \
  --sensor-max-soft-resets-per-window "${SENSOR_MAX_SOFT_RESETS_PER_WINDOW:-12}" \
  --sensor-recovery-min-healthy-probes "${SENSOR_RECOVERY_MIN_HEALTHY_PROBES:-2}" \
  --sensor-soft-reset-attempts "${SENSOR_SOFT_RESET_ATTEMPTS:-1}" \
  --sensor-controller-restart-attempts "${SENSOR_CONTROLLER_RESTART_ATTEMPTS:-2}" \
  --sensor-max-controller-restarts-per-hour "${SENSOR_MAX_CONTROLLER_RESTARTS_PER_HOUR:-6}" \
  --vision-backbone "${VISION_BACKBONE:-pixel-stats}" \
  --stop-on-mastery \
  --visual --open-browser "$@"
