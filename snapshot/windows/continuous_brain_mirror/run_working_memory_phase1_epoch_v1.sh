#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/continuous_brain_v0.12.2}"
cd "$PROJECT_ROOT"

STATE_DIR="${STATE_DIR:-state/working_memory_m1/train_epoch1_v1}"
LOG="$STATE_DIR/run.log"
UPDATES=1600

mkdir -p "$STATE_DIR"

RESUME_ARGS=()
if [ -f "$STATE_DIR/latest.portable.pt" ]; then
  RESUME_ARGS=(--resume "$STATE_DIR/latest.portable.pt")
  echo "resume_checkpoint=$STATE_DIR/latest.portable.pt"
else
  echo "resume_checkpoint=none"
fi

set +e
python train_working_memory_dml_v1.py \
  --updates "$UPDATES" \
  --state-dir "$STATE_DIR" \
  --lr 1e-4 \
  --schedule-seed 73191 \
  --init-seed 12345 \
  --decision-fraction 0.50 \
  --world-loss-weight 0.25 \
  --save-every 100 \
  --verify-every 50 \
  "${RESUME_ARGS[@]}" \
  2>&1 | tee -a "$LOG"
status=${PIPESTATUS[0]}
set -e

echo "training_exit=$status" | tee -a "$LOG"

if [ "$status" -ne 0 ]; then
  echo "WORKING_MEMORY_PHASE1: FAIL" | tee -a "$LOG"
  exit "$status"
fi

if grep -Eqi \
  'fall back to run on the CPU|falling back to CPU|CPU fallback' \
  "$LOG"
then
  echo "WORKING_MEMORY_PHASE1: FAIL_CPU_FALLBACK" | tee -a "$LOG"
  exit 3
fi

grep -q 'WORKING_MEMORY_TRAINING_COMPLETE' "$LOG"
grep -q 'memory_disabled_exact_base_parity: true' "$LOG"
grep -q 'portable_cpu_checkpoint: true' "$LOG"

echo "NO_REPORTED_CPU_FALLBACK" | tee -a "$LOG"
echo "WORKING_MEMORY_PHASE1: PASS" | tee -a "$LOG"
