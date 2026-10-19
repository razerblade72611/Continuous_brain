#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
cd "$PROJECT_ROOT"

echo "=== v0.37.0 Compositional Autobiographical Goals DEV ==="
echo "Frozen CPU models. No neural training. Requires closed v0.36.1 TEST PASS."
echo "Fresh DEV only. Historical results are retained; every run gets a new directory."

python -m py_compile \
  continuous_brain/compositional_autobiographical_goals_v1.py \
  evaluate_compositional_autobiographical_goals_v1.py \
  test_compositional_autobiographical_goals_v1.py
python test_compositional_autobiographical_goals_v1.py

set +e
python -u evaluate_compositional_autobiographical_goals_v1.py
STATUS=$?
set -e

case "$STATUS" in
  0) echo "COMPOSITIONAL_GOALS_WRAPPER: PASS_CONFIRMED" ;;
  4) echo "COMPOSITIONAL_GOALS_WRAPPER: DEV_FAIL_PRESERVED" ;;
  *) echo "COMPOSITIONAL_GOALS_WRAPPER: INFRASTRUCTURE_ERROR (not a scientific PASS/FAIL)" ;;
esac
exit "$STATUS"
