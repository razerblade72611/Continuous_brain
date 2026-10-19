#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/continuous_brain_v0.12.2}"
cd "$PROJECT_ROOT"

ROOT="state/working_memory_recall_smoke_v1"
UPDATES=25

mkdir -p "$ROOT"

for arm in linear gated; do
  STATE="$ROOT/$arm"
  mkdir -p "$STATE"
  LOG="$STATE/run.log"

  echo "=== production smoke arm=$arm ==="

  set +e
  python train_working_memory_recall_dml_v1.py \
    --arm "$arm" \
    --updates "$UPDATES" \
    --state-dir "$STATE" \
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

  echo "${arm}_training_exit=$status"

  if [ "$status" -ne 0 ]; then
    echo "WORKING_MEMORY_RECALL_SMOKE_RUN: FAIL_$arm"
    exit "$status"
  fi

  if grep -Eqi \
    'fall back to run on the CPU|falling back to CPU|CPU fallback' \
    "$LOG"
  then
    echo "WORKING_MEMORY_RECALL_SMOKE_RUN: FAIL_CPU_FALLBACK_$arm"
    exit 3
  fi

  grep -q 'memory_and_recall_disabled_exact_base_parity: true' "$LOG"
  grep -q 'portable_cpu_checkpoint: true' "$LOG"
  grep -q 'WORKING_MEMORY_RECALL_TRAINING_COMPLETE' "$LOG"

  echo "arm=$arm TRAINER_SMOKE_PASS"
done

python compare_working_memory_recall_smoke_v1.py \
  --linear "$ROOT/linear/training.jsonl" \
  --gated "$ROOT/gated/training.jsonl"

echo "NO_REPORTED_CPU_FALLBACK"
echo "WORKING_MEMORY_RECALL_SMOKE_RUN: PASS"
