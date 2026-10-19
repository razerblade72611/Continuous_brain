#!/usr/bin/env bash
set -euo pipefail

BASE_CHECKPOINT="${BASE_CHECKPOINT:-state/child_v113_dml_smoke/infant_training.pt}"
DEVICE="${DEVICE:-cpu}"
UPDATES="${UPDATES:-4000}"
EPISODES_PER_LENGTH="${EPISODES_PER_LENGTH:-100}"
SEED="${SEED:-81501}"
STATE_DIR="${STATE_DIR:-state/competitive_broadcast_v3}"

if [[ ! -f "$BASE_CHECKPOINT" ]]; then
  echo "Base checkpoint not found: $BASE_CHECKPOINT" >&2
  exit 2
fi

mkdir -p "$STATE_DIR"

# DirectML is qualified in fresh subprocesses because a DXGI device-removal
# failure poisons the current process.  Do not silently fall back to CPU.
case "${DEVICE,,}" in
  directml|dml)
    DEVICE="$DEVICE" CHECKPOINT="$BASE_CHECKPOINT" ./scripts/qualify_directml_v121.sh
    ;;
esac

python -m continuous_brain.test_cued_detour_world
python -m continuous_brain.test_competitive_infant

python -m continuous_brain.evaluate_competitive_infant train \
  --base-checkpoint "$BASE_CHECKPOINT" \
  --condition shared \
  --updates "$UPDATES" \
  --seed "$SEED" \
  --device "$DEVICE" \
  --state-dir "$STATE_DIR"

python -m continuous_brain.evaluate_competitive_infant train \
  --base-checkpoint "$BASE_CHECKPOINT" \
  --condition independent \
  --updates "$UPDATES" \
  --seed "$SEED" \
  --device "$DEVICE" \
  --state-dir "$STATE_DIR"

python -m continuous_brain.evaluate_competitive_infant train \
  --base-checkpoint "$BASE_CHECKPOINT" \
  --condition uniform \
  --updates "$UPDATES" \
  --seed "$SEED" \
  --device "$DEVICE" \
  --state-dir "$STATE_DIR"

python -m continuous_brain.evaluate_competitive_infant evaluate \
  --shared-checkpoint "$STATE_DIR/competitive_infant_shared.pt" \
  --independent-checkpoint "$STATE_DIR/competitive_infant_independent.pt" \
  --uniform-checkpoint "$STATE_DIR/competitive_infant_shared_uniform.pt" \
  --episodes-per-length "$EPISODES_PER_LENGTH" \
  --device "$DEVICE" \
  --output "$STATE_DIR/competitive_broadcast_results.json"

echo
echo "Finished. Results: $STATE_DIR/competitive_broadcast_results.json"
