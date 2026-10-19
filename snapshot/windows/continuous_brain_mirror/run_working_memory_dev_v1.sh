#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/continuous_brain_v0.12.2}"
cd "$PROJECT_ROOT"

STATE_ROOT="state/working_memory_m1"
CHECKPOINT="$STATE_ROOT/train_epoch1_v1/latest.portable.pt"
EVAL_JSON="$STATE_ROOT/dev_eval_v1.json"
EVAL_LOG="$STATE_ROOT/dev_eval_v1.log"
DECISION_JSON="$STATE_ROOT/dev_decision_v1.json"
DECISION_LOG="$STATE_ROOT/dev_decision_v1.log"

if [ ! -f "$CHECKPOINT" ]; then
  echo "missing checkpoint: $CHECKPOINT"
  exit 2
fi

set +e
python evaluate_working_memory_dev_v1.py \
  --checkpoint "$CHECKPOINT" \
  --output "$EVAL_JSON" \
  2>&1 | tee "$EVAL_LOG"
eval_status=${PIPESTATUS[0]}
set -e

echo "dev_eval_exit=$eval_status"
if [ "$eval_status" -ne 0 ]; then
  echo "WORKING_MEMORY_DEV_RUN: FAIL_EVALUATOR"
  exit "$eval_status"
fi

if grep -Eqi \
  'fall back to run on the CPU|falling back to CPU|CPU fallback' \
  "$EVAL_LOG"
then
  echo "WORKING_MEMORY_DEV_RUN: FAIL_CPU_FALLBACK"
  exit 3
fi

python decide_working_memory_dev_v1.py \
  --input "$EVAL_JSON" \
  --output "$DECISION_JSON" \
  2>&1 | tee "$DECISION_LOG"

echo "NO_REPORTED_CPU_FALLBACK"
echo "WORKING_MEMORY_DEV_RUN_COMPLETE"
