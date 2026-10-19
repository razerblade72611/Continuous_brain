#!/usr/bin/env bash
set -euo pipefail

AUDIT_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
AUDIT_PROJECT_ROOT="${PROJECT_ROOT:-$AUDIT_SCRIPT_DIR}"
export PYTHONDONTWRITEBYTECODE=1
exec python "$AUDIT_SCRIPT_DIR/audit_planning_option_integration_v1_1.py" \
  --project-root "$AUDIT_PROJECT_ROOT" "$@"
