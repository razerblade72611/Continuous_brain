#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/continuous_brain_v0.12.2}"
cd "$PROJECT_ROOT"

STATE="state/working_memory_recall_capacity_control_qualification_v1"
mkdir -p "$STATE"
LOG="$STATE/run.log"

set +e
python qualify_working_memory_recall_capacity_control_v1.py \
  2>&1 | tee "$LOG"
status=${PIPESTATUS[0]}
set -e

echo "qualification_exit=$status"

if [ "$status" -ne 0 ]; then
  echo "WORKING_MEMORY_CAPACITY_CONTROL_QUALIFICATION_RUN: FAIL"
  exit "$status"
fi

if grep -Eqi \
  'fall back to run on the CPU|falling back to CPU|CPU fallback' \
  "$LOG"
then
  echo "WORKING_MEMORY_CAPACITY_CONTROL_QUALIFICATION_RUN: FAIL_CPU_FALLBACK"
  exit 3
fi

grep -q 'recall_parameters: 49280' "$LOG"
grep -q 'total_addition_parameters: 86272' "$LOG"
grep -q 'contextual_hidden_input: false' "$LOG"
grep -q 'recall_bias_terms: false' "$LOG"
grep -q 'EXACT_INITIAL_NOOP_PAIR_OK' "$LOG"
grep -q 'historical_base_bit_exact: true' "$LOG"
grep -q 'staged_capacity_control_gradients: true' "$LOG"
grep -q 'learned_recall_nonzero: true' "$LOG"
grep -q 'WORKING_MEMORY_CAPACITY_CONTROL_QUALIFICATION_OK' "$LOG"

echo "NO_REPORTED_CPU_FALLBACK"
echo "WORKING_MEMORY_CAPACITY_CONTROL_QUALIFICATION_RUN: PASS"
