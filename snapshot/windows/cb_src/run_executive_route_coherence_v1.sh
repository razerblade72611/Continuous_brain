#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/continuous_brain_v0.12.2}"
cd "$PROJECT_ROOT"

ROOT="state/executive_novel_rules_v2"
mkdir -p "$ROOT"

python diagnose_executive_route_coherence_v1.py \
  --output "$ROOT/route_coherence_diagnostic_v1.json" \
  2>&1 | tee "$ROOT/route_coherence_diagnostic_v1.log"

status=${PIPESTATUS[0]}
if [ "$status" -ne 0 ]; then
  echo "EXECUTIVE_ROUTE_COHERENCE_RUN: FAIL"
  exit "$status"
fi

if grep -Eqi \
  'fall back to run on the CPU|falling back to CPU|CPU fallback' \
  "$ROOT/route_coherence_diagnostic_v1.log"
then
  echo "EXECUTIVE_ROUTE_COHERENCE_RUN: FAIL_CPU_FALLBACK"
  exit 3
fi

grep -q 'admitted_brain_bit_exact: true' \
  "$ROOT/route_coherence_diagnostic_v1.log"
grep -q 'historical_base_bit_exact: true' \
  "$ROOT/route_coherence_diagnostic_v1.log"
grep -q 'FRESH_TEST_TOUCHED: false' \
  "$ROOT/route_coherence_diagnostic_v1.log"
grep -q 'EXECUTIVE_ROUTE_COHERENCE_DIAGNOSTIC_COMPLETE' \
  "$ROOT/route_coherence_diagnostic_v1.log"

echo "NO_REPORTED_CPU_FALLBACK"
echo "NO_TRAINING_PERFORMED: true"
echo "FRESH_TEST_REMAINS_SEALED: true"
echo "EXECUTIVE_ROUTE_COHERENCE_RUN: PASS"
