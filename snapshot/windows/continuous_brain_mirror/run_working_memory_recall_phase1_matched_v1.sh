#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/continuous_brain_v0.12.2}"
cd "$PROJECT_ROOT"

ROOT="state/working_memory_recall_phase1_v1"
UPDATES=1600
LR="1e-4"
SCHEDULE_SEED=73191
INIT_SEED=12345
DECISION_FRACTION="0.50"
WORLD_LOSS_WEIGHT="0.25"

mkdir -p "$ROOT"

for arm in linear gated; do
  STATE="$ROOT/$arm"
  mkdir -p "$STATE"

  RESUME_ARGS=()
  if [ -f "$STATE/latest.portable.pt" ]; then
    echo "arm=$arm resume_checkpoint=$STATE/latest.portable.pt"
    RESUME_ARGS=(--resume "$STATE/latest.portable.pt")
  else
    echo "arm=$arm resume_checkpoint=none"
  fi

  ATTEMPT_LOG="$STATE/attempt_$(date +%Y%m%d_%H%M%S).log"

  set +e
  python train_working_memory_recall_dml_v2.py \
    --arm "$arm" \
    --updates "$UPDATES" \
    --state-dir "$STATE" \
    --lr "$LR" \
    --schedule-seed "$SCHEDULE_SEED" \
    --init-seed "$INIT_SEED" \
    --decision-fraction "$DECISION_FRACTION" \
    --world-loss-weight "$WORLD_LOSS_WEIGHT" \
    --save-every 100 \
    --verify-every 50 \
    "${RESUME_ARGS[@]}" \
    2>&1 | tee "$ATTEMPT_LOG"
  status=${PIPESTATUS[0]}
  set -e

  echo "${arm}_training_exit=$status"

  if [ "$status" -ne 0 ]; then
    echo "WORKING_MEMORY_RECALL_PHASE1_RUN: FAIL_$arm"
    exit "$status"
  fi

  if grep -Eqi \
    'fall back to run on the CPU|falling back to CPU|CPU fallback' \
    "$ATTEMPT_LOG"
  then
    echo "WORKING_MEMORY_RECALL_PHASE1_RUN: FAIL_CPU_FALLBACK_$arm"
    exit 3
  fi

  grep -q \
    'base_digest_final: 8de305f12369f08056c8ef607b2aaae3e57b620c882a9216fe11a4b48bb5822e' \
    "$ATTEMPT_LOG"
  grep -q 'memory_and_recall_disabled_exact_base_parity: true' "$ATTEMPT_LOG"
  grep -q 'portable_cpu_checkpoint: true' "$ATTEMPT_LOG"
  grep -q 'WORKING_MEMORY_RECALL_TRAINING_COMPLETE' "$ATTEMPT_LOG"

  echo "arm=$arm PHASE1_TRAIN_PASS"
done

M1_ARGS=()
if [ -f "state/working_memory_m1/train_epoch1_v1/training.jsonl" ]; then
  M1_ARGS=(
    --m1-jsonl
    "state/working_memory_m1/train_epoch1_v1/training.jsonl"
  )
fi

python audit_working_memory_recall_phase1_v1.py \
  --linear "$ROOT/linear/training.jsonl" \
  --gated "$ROOT/gated/training.jsonl" \
  "${M1_ARGS[@]}" \
  --output "$ROOT/phase1_audit.json" \
  | tee "$ROOT/phase1_audit.log"

echo "NO_REPORTED_CPU_FALLBACK"
echo "WORKING_MEMORY_RECALL_PHASE1_RUN: PASS"
