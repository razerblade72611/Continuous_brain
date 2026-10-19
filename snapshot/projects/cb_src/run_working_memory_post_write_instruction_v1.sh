#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/continuous_brain_v0.12.2}"
cd "$PROJECT_ROOT"

OUT="state/working_memory_recall_capacity_control_phase1_v1"
LOG="$OUT/post_write_instruction_probe_v1.log"

python probe_working_memory_post_write_instruction_v1.py \
  --checkpoint "$OUT/latest.portable.pt" \
  --output "$OUT/post_write_instruction_probe_v1.json" \
  2>&1 | tee "$LOG"

status=${PIPESTATUS[0]}
if [ "$status" -ne 0 ]; then
  echo "WORKING_MEMORY_POST_WRITE_INSTRUCTION_RUN: FAIL"
  exit "$status"
fi

if grep -Eqi \
  'fall back to run on the CPU|falling back to CPU|CPU fallback' \
  "$LOG"
then
  echo "WORKING_MEMORY_POST_WRITE_INSTRUCTION_RUN: FAIL_BRAIN_CPU_FALLBACK"
  exit 3
fi

grep -q 'writer_frozen: true' "$LOG"
grep -q 'instruction_exists_during_brain_rollout: false' "$LOG"
grep -q 'instruction_generated_only_after_memory_collection: true' "$LOG"
grep -q 'POST_WRITE_COMPOSITION_GATE: PASS' "$LOG"
grep -q 'DEV_TOUCHED: false' "$LOG"
grep -q 'TEST_TOUCHED: false' "$LOG"
grep -q 'WORKING_MEMORY_POST_WRITE_INSTRUCTION_PROBE_COMPLETE' "$LOG"

echo "BRAIN_NO_REPORTED_CPU_FALLBACK"
echo "NEW_READER_CPU_IS_INTENTIONAL: true"
echo "WORKING_MEMORY_POST_WRITE_INSTRUCTION_RUN: PASS"
