#!/usr/bin/env bash
set -euo pipefail
# The official Meta DINOv3 weights are gated on Hugging Face. Accept the model
# license and authenticate before first use. On CPU, --vision-every 5 reduces cost.
continuous-brain --state-dir state/my_brain --visual --vision-backbone dinov3-vits16 --vision-every 5 "$@"
