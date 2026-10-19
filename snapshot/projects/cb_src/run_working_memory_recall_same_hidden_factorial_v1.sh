#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/continuous_brain_v0.12.2}"
cd "$PROJECT_ROOT"

OUT="state/working_memory_recall_phase1_v1/final_train_causal_v1/same_hidden_factorial_v1"
mkdir -p "$OUT"

for arm in linear gated; do
  CKPT="state/working_memory_recall_phase1_v1/$arm/latest.portable.pt"
  LOG="$OUT/$arm.log"
  JSON="$OUT/$arm.json"

  set +e
  python probe_working_memory_recall_same_hidden_factorial_v1.py \
    --arm "$arm" \
    --checkpoint "$CKPT" \
    --output "$JSON" \
    2>&1 | tee "$LOG"
  status=${PIPESTATUS[0]}
  set -e

  if [ "$status" -ne 0 ]; then
    echo "SAME_HIDDEN_FACTORIAL_RUN: FAIL_$arm"
    exit "$status"
  fi

  if grep -Eqi \
    'fall back to run on the CPU|falling back to CPU|CPU fallback' \
    "$LOG"
  then
    echo "SAME_HIDDEN_FACTORIAL_RUN: FAIL_CPU_FALLBACK_$arm"
    exit 3
  fi

  grep -q 'historical_base_bit_exact: true' "$LOG"
  grep -q 'DEV_TOUCHED: false' "$LOG"
  grep -q 'TEST_TOUCHED: false' "$LOG"
  grep -q 'WORKING_MEMORY_RECALL_SAME_HIDDEN_FACTORIAL_COMPLETE' "$LOG"
done

echo "NO_REPORTED_CPU_FALLBACK"
echo "SAME_HIDDEN_FACTORIAL_RUN: PASS"
