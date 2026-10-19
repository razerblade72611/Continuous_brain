#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/continuous_brain_v0.12.2}"
cd "$PROJECT_ROOT"

ROOT="state/executive_composition_phase1_v1"
mkdir -p "$ROOT"

echo "=== Executive Phase 1: task audit ==="
python audit_late_instruction_task_v1.py \
  2>&1 | tee "$ROOT/task_audit.log"

grep -q 'LATE_INSTRUCTION_TASK_AUDIT_V1: PASS' "$ROOT/task_audit.log"
grep -q 'OLD_DEV_IDS_REJECTED: true' "$ROOT/task_audit.log"
grep -q 'OLD_TEST_IDS_REJECTED: true' "$ROOT/task_audit.log"

echo "=== Executive Phase 1: neural qualification ==="
python qualify_executive_composition_v1.py \
  2>&1 | tee "$ROOT/qualification.log"

if grep -Eqi \
  'fall back to run on the CPU|falling back to CPU|CPU fallback' \
  "$ROOT/qualification.log"
then
  echo "EXECUTIVE_COMPOSITION_PHASE1_RUN: FAIL_CPU_FALLBACK_QUALIFICATION"
  exit 3
fi

grep -q 'EXACT_INITIAL_EXECUTIVE_NOOP_OK' "$ROOT/qualification.log"
grep -q 'admitted_checkpoint_unchanged: true' "$ROOT/qualification.log"
grep -q 'admitted_brain_bit_exact: true' "$ROOT/qualification.log"
grep -q 'historical_base_bit_exact: true' "$ROOT/qualification.log"
grep -q 'instruction_none_exact_noop_after_training: true' "$ROOT/qualification.log"
grep -q 'staged_executive_gradients: true' "$ROOT/qualification.log"
grep -q 'EXECUTIVE_COMPOSITION_QUALIFICATION_OK' "$ROOT/qualification.log"

echo "=== Executive Phase 1: 50-update balanced learning smoke ==="

SMOKE_STATE="$ROOT/smoke_50_v1"
if [ -f "$SMOKE_STATE/training.jsonl" ]; then
  echo "refusing to overwrite existing smoke state: $SMOKE_STATE"
  echo "move or remove that directory only if you intentionally want a fresh rerun"
  exit 4
fi

python train_executive_composition_smoke_v1.py \
  --updates 50 \
  --state-dir "$SMOKE_STATE" \
  --lr 2e-4 \
  --schedule-seed 92731 \
  --init-seed 24680 \
  2>&1 | tee "$ROOT/smoke.log"

if grep -Eqi \
  'fall back to run on the CPU|falling back to CPU|CPU fallback' \
  "$ROOT/smoke.log"
then
  echo "EXECUTIVE_COMPOSITION_PHASE1_RUN: FAIL_CPU_FALLBACK_SMOKE"
  exit 5
fi

grep -q 'portable_cpu_checkpoint: true' "$ROOT/smoke.log"
grep -q 'admitted_checkpoint_unchanged: true' "$ROOT/smoke.log"
grep -q 'admitted_brain_bit_exact: true' "$ROOT/smoke.log"
grep -q 'historical_base_bit_exact: true' "$ROOT/smoke.log"
grep -q 'OLD_DEV_TOUCHED: false' "$ROOT/smoke.log"
grep -q 'OLD_TEST_TOUCHED: false' "$ROOT/smoke.log"
grep -q 'EXECUTIVE_COMPOSITION_LEARNING_SMOKE_COMPLETE' "$ROOT/smoke.log"

echo "NO_REPORTED_CPU_FALLBACK"
echo "OLD_DEV_REUSED_FOR_EXECUTIVE: false"
echo "OLD_TEST_REUSED_FOR_EXECUTIVE: false"
echo "EXECUTIVE_COMPOSITION_PHASE1_RUN: PASS"
