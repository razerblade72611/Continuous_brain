#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m continuous_brain.individual_cli \
  --environment development --profile "${PROFILE:-standard}" \
  --state-dir "${STATE_DIR:-state/individual_v062}" \
  --vision-backbone "${VISION_BACKBONE:-pixel-stats}" \
  --visual --open-browser "$@"
