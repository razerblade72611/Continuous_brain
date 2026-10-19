#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/continuous_brain_v0.12.2}"
cd "$PROJECT_ROOT"

STATE="state/executive_composition_production_v1"
mkdir -p "$STATE"

if [ -f "$STATE/training.jsonl" ]; then
  echo "existing production state detected: $STATE"
  echo "This v1 runner intentionally refuses an ambiguous restart."
  echo "If you are resuming a prior interrupted run, stop here and send me the state/log."
  exit 2
fi

echo "=== Executive Composition Production v1 ==="
echo "TRAIN only: conditions 0..1599"
echo "Old memory DEV is not reused."
echo "Old TEST remains sealed."

python train_executive_composition_production_v1.py \
  --updates 1600 \
  --state-dir "$STATE" \
  --lr 2e-4 \
  --schedule-seed 92731 \
  --init-seed 24680 \
  2>&1 | tee "$STATE/run.log"

if grep -Eqi \
  'fall back to run on the CPU|falling back to CPU|CPU fallback' \
  "$STATE/run.log"
then
  echo "EXECUTIVE_COMPOSITION_PRODUCTION_RUN: FAIL_CPU_FALLBACK"
  exit 3
fi

grep -q 'EXECUTIVE_COMPOSITION_PRODUCTION_TRAINING_COMPLETE' "$STATE/run.log"
grep -q 'admitted_checkpoint_unchanged: true' "$STATE/run.log"
grep -q 'admitted_brain_bit_exact: true' "$STATE/run.log"
grep -q 'historical_base_bit_exact: true' "$STATE/run.log"
grep -q 'OLD_DEV_TOUCHED: false' "$STATE/run.log"
grep -q 'OLD_TEST_TOUCHED: false' "$STATE/run.log"

CKPT="$STATE/latest.portable.pt"
if [ ! -f "$CKPT" ]; then
  echo "missing final production checkpoint"
  exit 4
fi

echo "=== Fixed-final TRAIN causal evaluation ==="

python evaluate_executive_composition_final_train_causal_v1.py \
  --checkpoint "$CKPT" \
  --output "$STATE/final_train_causal_v1.json" \
  2>&1 | tee "$STATE/final_train_causal_v1.log"

if grep -Eqi \
  'fall back to run on the CPU|falling back to CPU|CPU fallback' \
  "$STATE/final_train_causal_v1.log"
then
  echo "EXECUTIVE_COMPOSITION_PRODUCTION_RUN: FAIL_EVAL_CPU_FALLBACK"
  exit 5
fi

grep -q 'admitted_brain_bit_exact: true' "$STATE/final_train_causal_v1.log"
grep -q 'historical_base_bit_exact: true' "$STATE/final_train_causal_v1.log"
grep -q 'OLD_DEV_TOUCHED: false' "$STATE/final_train_causal_v1.log"
grep -q 'OLD_TEST_TOUCHED: false' "$STATE/final_train_causal_v1.log"
grep -q 'EXECUTIVE_COMPOSITION_FINAL_TRAIN_CAUSAL_COMPLETE' "$STATE/final_train_causal_v1.log"

set +e
python decide_executive_composition_train_learning_v1.py \
  --input "$STATE/final_train_causal_v1.json" \
  --output "$STATE/train_learning_decision_v1.json" \
  2>&1 | tee "$STATE/train_learning_decision_v1.log"
status=${PIPESTATUS[0]}
set -e

echo "decision_exit=$status"
echo "OLD_DEV_REUSED_FOR_EXECUTIVE: false"
echo "OLD_TEST_REUSED_FOR_EXECUTIVE: false"

if [ "$status" -eq 0 ]; then
  echo "EXECUTIVE_COMPOSITION_PRODUCTION_RUN: PASS"
else
  echo "EXECUTIVE_COMPOSITION_PRODUCTION_RUN: FAIL_LEARNING_GATE"
fi

exit "$status"
