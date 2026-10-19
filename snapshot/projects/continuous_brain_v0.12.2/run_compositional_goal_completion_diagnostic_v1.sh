#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
cd "$PROJECT_ROOT"

echo "=== v0.37.1 Compositional Goal Completion Diagnostic ==="
echo "Read-only audit and frozen replay of the existing DEV run. No admission or training."
python -m py_compile \
  diagnose_compositional_goal_completion_v1.py \
  test_compositional_goal_completion_diagnostic_v1.py
python test_compositional_goal_completion_diagnostic_v1.py

set +e
python -u diagnose_compositional_goal_completion_v1.py "$@"
STATUS=$?
set -e

if [ "$STATUS" -eq 0 ]; then
  echo "COMPOSITIONAL_COMPLETION_DIAGNOSTIC_WRAPPER: COMPLETE (v0.37.0 remains DEV FAIL)"
else
  echo "COMPOSITIONAL_COMPLETION_DIAGNOSTIC_WRAPPER: INFRASTRUCTURE_ERROR"
fi
exit "$STATUS"
