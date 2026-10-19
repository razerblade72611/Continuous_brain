#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/continuous_brain_v0.12.2}"
cd "$PROJECT_ROOT"
ROOT="state/working_memory_recall_phase1_v1"
OUT="$ROOT/final_train_causal_v1"
mkdir -p "$OUT"
for arm in linear gated; do
  CKPT="$ROOT/$arm/latest.portable.pt"
  JSON="$OUT/$arm.json"
  LOG="$OUT/$arm.log"
  [ -f "$CKPT" ] || { echo "missing checkpoint: $CKPT"; exit 2; }
  echo "=== fixed final TRAIN causal eval arm=$arm ==="
  set +e
  python evaluate_working_memory_recall_train_causal_v1.py --arm "$arm" --checkpoint "$CKPT" --output "$JSON" 2>&1 | tee "$LOG"
  status=${PIPESTATUS[0]}
  set -e
  echo "${arm}_causal_eval_exit=$status"
  [ "$status" -eq 0 ] || { echo "WORKING_MEMORY_RECALL_TRAIN_CAUSAL_RUN: FAIL_$arm"; exit "$status"; }
  if grep -Eqi 'fall back to run on the CPU|falling back to CPU|CPU fallback' "$LOG"; then
    echo "WORKING_MEMORY_RECALL_TRAIN_CAUSAL_RUN: FAIL_CPU_FALLBACK_$arm"; exit 3
  fi
  grep -q 'historical_base_bit_exact: true' "$LOG"
  grep -q 'DEV_TOUCHED: false' "$LOG"
  grep -q 'TEST_TOUCHED: false' "$LOG"
  grep -q 'WORKING_MEMORY_RECALL_TRAIN_CAUSAL_EVAL_COMPLETE' "$LOG"
  echo "arm=$arm FINAL_TRAIN_CAUSAL_PASS"
done
python compare_working_memory_recall_final_train_v1.py --linear "$OUT/linear.json" --gated "$OUT/gated.json" | tee "$OUT/comparison.log"
echo "NO_REPORTED_CPU_FALLBACK"
echo "WORKING_MEMORY_RECALL_TRAIN_CAUSAL_RUN: PASS"
