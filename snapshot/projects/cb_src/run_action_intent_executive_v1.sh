#!/usr/bin/env bash
# Compatibility entrypoint: the original command now uses measured checks.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
exec python -u run_action_intent_repair_v1.py "$@"
