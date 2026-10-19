#!/usr/bin/env bash
set -euo pipefail
python -m continuous_brain.cli --state-dir state/demo --brain-hz 10 "$@"
