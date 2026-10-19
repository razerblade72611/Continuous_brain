#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/continuous_brain_v0.12.2}"
cd "$PROJECT_ROOT"

STATE="state/working_memory_recall_capacity_control_phase1_v1"
UPDATES=1600
mkdir -p "$STATE"

RESUME_ARGS=()
if [ -f "$STATE/latest.portable.pt" ]; then
  echo "resume_checkpoint=$STATE/latest.portable.pt"
  RESUME_ARGS=(--resume "$STATE/latest.portable.pt")
else
  echo "resume_checkpoint=none"
fi

ATTEMPT_LOG="$STATE/attempt_$(date +%Y%m%d_%H%M%S).log"

set +e
python train_working_memory_recall_capacity_control_dml_v1.py \
  --arm capacity_ungated \
  --updates "$UPDATES" \
  --state-dir "$STATE" \
  --lr 1e-4 \
  --schedule-seed 73191 \
  --init-seed 12345 \
  --decision-fraction 0.50 \
  --world-loss-weight 0.25 \
  --save-every 100 \
  --verify-every 50 \
  "${RESUME_ARGS[@]}" \
  2>&1 | tee "$ATTEMPT_LOG"
status=${PIPESTATUS[0]}
set -e

echo "capacity_training_exit=$status"
if [ "$status" -ne 0 ]; then
  echo "WORKING_MEMORY_CAPACITY_CONTROL_PHASE1_RUN: FAIL_TRAIN"
  exit "$status"
fi

if grep -Eqi \
  'fall back to run on the CPU|falling back to CPU|CPU fallback' \
  "$ATTEMPT_LOG"
then
  echo "WORKING_MEMORY_CAPACITY_CONTROL_PHASE1_RUN: FAIL_CPU_FALLBACK"
  exit 3
fi

grep -q \
  'base_digest_final: 8de305f12369f08056c8ef607b2aaae3e57b620c882a9216fe11a4b48bb5822e' \
  "$ATTEMPT_LOG"
grep -q 'memory_and_recall_disabled_exact_base_parity: true' "$ATTEMPT_LOG"
grep -q 'portable_cpu_checkpoint: true' "$ATTEMPT_LOG"
grep -q 'WORKING_MEMORY_CAPACITY_CONTROL_TRAINING_COMPLETE' "$ATTEMPT_LOG"

echo "CAPACITY_CONTROL_PHASE1_TRAIN_PASS"

CAUSAL_DIR="$STATE/final_train_causal_v1"
mkdir -p "$CAUSAL_DIR"

set +e
python evaluate_working_memory_recall_capacity_control_train_causal_v1.py \
  --checkpoint "$STATE/latest.portable.pt" \
  --output "$CAUSAL_DIR/capacity_ungated.json" \
  2>&1 | tee "$CAUSAL_DIR/capacity_ungated.log"
status=${PIPESTATUS[0]}
set -e

echo "capacity_causal_exit=$status"
if [ "$status" -ne 0 ]; then
  echo "WORKING_MEMORY_CAPACITY_CONTROL_PHASE1_RUN: FAIL_CAUSAL"
  exit "$status"
fi

if grep -Eqi \
  'fall back to run on the CPU|falling back to CPU|CPU fallback' \
  "$CAUSAL_DIR/capacity_ungated.log"
then
  echo "WORKING_MEMORY_CAPACITY_CONTROL_PHASE1_RUN: FAIL_CAUSAL_CPU_FALLBACK"
  exit 4
fi

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
  exit 5
fi

python compare_working_memory_recall_three_way_final_train_v1.py \
  --linear "$LINEAR_JSON" \
  --control "$CAUSAL_DIR/capacity_ungated.json" \
  --gated "$GATED_JSON" \
  | tee "$CAUSAL_DIR/three_way_comparison.log"

echo "NO_REPORTED_CPU_FALLBACK"
echo "WORKING_MEMORY_CAPACITY_CONTROL_PHASE1_RUN: PASS"
