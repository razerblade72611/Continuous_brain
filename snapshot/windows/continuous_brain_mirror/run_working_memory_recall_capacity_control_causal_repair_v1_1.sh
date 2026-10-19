#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/continuous_brain_v0.12.2}"
cd "$PROJECT_ROOT"

STATE="state/working_memory_recall_capacity_control_phase1_v1"
CKPT="$STATE/latest.portable.pt"
CAUSAL_DIR="$STATE/final_train_causal_v1"
mkdir -p "$CAUSAL_DIR"

if [ ! -f "$CKPT" ]; then
  echo "missing completed capacity-control checkpoint: $CKPT"
  exit 2
fi

echo "REUSING_COMPLETED_TRAINING_CHECKPOINT: $CKPT"
echo "NO_TRAINING_OR_OPTIMIZER_STEPS_WILL_RUN"

set +e
python evaluate_working_memory_recall_capacity_control_train_causal_v1.py \
  --checkpoint "$CKPT" \
  --output "$CAUSAL_DIR/capacity_ungated.json" \
  2>&1 | tee "$CAUSAL_DIR/capacity_ungated.log"
status=${PIPESTATUS[0]}
set -e

echo "capacity_causal_exit=$status"
if [ "$status" -ne 0 ]; then
  echo "WORKING_MEMORY_CAPACITY_CONTROL_CAUSAL_REPAIR_RUN: FAIL_CAUSAL"
  exit "$status"
fi

if grep -Eqi \
  'fall back to run on the CPU|falling back to CPU|CPU fallback' \
  "$CAUSAL_DIR/capacity_ungated.log"
then
  echo "WORKING_MEMORY_CAPACITY_CONTROL_CAUSAL_REPAIR_RUN: FAIL_CPU_FALLBACK"
  exit 3
fi

grep -q 'arm: capacity_ungated' "$CAUSAL_DIR/capacity_ungated.log"
grep -q 'checkpoint_update: 1600' "$CAUSAL_DIR/capacity_ungated.log"
grep -q 'historical_base_bit_exact: true' "$CAUSAL_DIR/capacity_ungated.log"
grep -q 'DEV_TOUCHED: false' "$CAUSAL_DIR/capacity_ungated.log"
grep -q 'TEST_TOUCHED: false' "$CAUSAL_DIR/capacity_ungated.log"
grep -q \
  'WORKING_MEMORY_CAPACITY_CONTROL_TRAIN_CAUSAL_EVAL_COMPLETE' \
  "$CAUSAL_DIR/capacity_ungated.log"

LINEAR_JSON="state/working_memory_recall_phase1_v1/final_train_causal_v1/linear.json"
GATED_JSON="state/working_memory_recall_phase1_v1/final_train_causal_v1/gated.json"

if [ ! -f "$LINEAR_JSON" ] || [ ! -f "$GATED_JSON" ]; then
  echo "missing existing M1.5/M2 final causal JSON"
  exit 4
fi

python compare_working_memory_recall_three_way_final_train_v1.py \
  --linear "$LINEAR_JSON" \
  --control "$CAUSAL_DIR/capacity_ungated.json" \
  --gated "$GATED_JSON" \
  | tee "$CAUSAL_DIR/three_way_comparison.log"

echo "NO_REPORTED_CPU_FALLBACK"
echo "TRAINING_REUSED_WITHOUT_MODIFICATION: true"
echo "DEV_TOUCHED: false"
echo "TEST_TOUCHED: false"
echo "WORKING_MEMORY_CAPACITY_CONTROL_CAUSAL_REPAIR_RUN: PASS"
