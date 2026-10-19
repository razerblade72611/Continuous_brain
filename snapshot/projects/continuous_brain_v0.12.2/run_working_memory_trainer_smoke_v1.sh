#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/continuous_brain_v0.12.2}"
cd "$PROJECT_ROOT"

STATE_DIR="${STATE_DIR:-state/working_memory_m1/smoke_v1}"
UPDATES="${UPDATES:-25}"

mkdir -p "$STATE_DIR"
LOG="$STATE_DIR/run.log"

set +e
python train_working_memory_dml_v1.py \
  --updates "$UPDATES" \
  --state-dir "$STATE_DIR" \
  --lr 1e-4 \
  --schedule-seed 73191 \
  --init-seed 12345 \
  --decision-fraction 0.50 \
  --world-loss-weight 0.25 \
  --save-every "$UPDATES" \
  --verify-every 5 \
  2>&1 | tee "$LOG"
status=${PIPESTATUS[0]}
set -e

echo "training_exit=$status"
if [ "$status" -ne 0 ]; then
  echo "WORKING_MEMORY_TRAINER_SMOKE: FAIL"
  exit "$status"
fi

if grep -Eqi \
  'fall back to run on the CPU|falling back to CPU|CPU fallback' \
  "$LOG"
then
  echo "WORKING_MEMORY_TRAINER_SMOKE: FAIL_CPU_FALLBACK"
  exit 3
fi

grep -q 'WORKING_MEMORY_TRAINING_COMPLETE' "$LOG"

echo "NO_REPORTED_CPU_FALLBACK"
echo "WORKING_MEMORY_TRAINER_SMOKE: PASS"
