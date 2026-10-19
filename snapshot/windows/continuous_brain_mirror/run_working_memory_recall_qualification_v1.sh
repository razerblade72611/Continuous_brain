#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/continuous_brain_v0.12.2}"
cd "$PROJECT_ROOT"

STATE_ROOT="state/working_memory_recall_qualification_v1"
mkdir -p "$STATE_ROOT"

for arm in linear gated; do
  LOG="$STATE_ROOT/${arm}.log"
  echo "=== qualifying arm=$arm ==="

  set +e
  python qualify_working_memory_recall_arm_v1.py \
    --arm "$arm" \
    2>&1 | tee "$LOG"
  status=${PIPESTATUS[0]}
  set -e

  echo "${arm}_qualification_exit=$status"

  if [ "$status" -ne 0 ]; then
    echo "WORKING_MEMORY_RECALL_QUALIFICATION_RUN: FAIL_$arm"
    exit "$status"
  fi

  if grep -Eqi \
    'fall back to run on the CPU|falling back to CPU|CPU fallback' \
    "$LOG"
  then
    echo "WORKING_MEMORY_RECALL_QUALIFICATION_RUN: FAIL_CPU_FALLBACK_$arm"
    exit 3
  fi

  grep -q 'EXACT_INITIAL_NOOP_PAIR_OK' "$LOG"
  grep -q 'historical_base_bit_exact: true' "$LOG"
  grep -q 'staged_recall_gradients: true' "$LOG"
  grep -q 'learned_recall_nonzero: true' "$LOG"
  grep -q 'WORKING_MEMORY_RECALL_QUALIFICATION_OK' "$LOG"

  echo "arm=$arm PASS"
done

echo "NO_REPORTED_CPU_FALLBACK"
echo "WORKING_MEMORY_RECALL_QUALIFICATION_RUN: PASS"
