#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/continuous_brain_v0.12.2}"
cd "$PROJECT_ROOT"

OUT="state/working_memory_recall_capacity_control_phase1_v1"
LOG="$OUT/memory_shuffle_mechanism_v1.log"

python diagnose_working_memory_shuffle_mechanism_v1.py \
  --checkpoint "$OUT/latest.portable.pt" \
  --output "$OUT/memory_shuffle_mechanism_v1.json" \
  2>&1 | tee "$LOG"

status=${PIPESTATUS[0]}
if [ "$status" -ne 0 ]; then
  echo "WORKING_MEMORY_SHUFFLE_MECHANISM_RUN: FAIL"
  exit "$status"
fi

if grep -Eqi \
  'fall back to run on the CPU|falling back to CPU|CPU fallback' \
  "$LOG"
then
  echo "WORKING_MEMORY_SHUFFLE_MECHANISM_RUN: FAIL_BRAIN_CPU_FALLBACK"
  exit 3
fi

grep -q 'original_control_shuffle_is_per_example: true' "$LOG"
grep -q 'purpose: mechanism_not_null_pvalue' "$LOG"
grep -q 'historical_base_bit_exact: true' "$LOG"
grep -q 'DEV_TOUCHED: false' "$LOG"
grep -q 'TEST_TOUCHED: false' "$LOG"
grep -q 'WORKING_MEMORY_SHUFFLE_MECHANISM_DIAGNOSTIC_COMPLETE' "$LOG"

echo "BRAIN_NO_REPORTED_CPU_FALLBACK"
echo "PROBE_CPU_IS_INTENTIONAL: true"
echo "WORKING_MEMORY_SHUFFLE_MECHANISM_RUN: PASS"
