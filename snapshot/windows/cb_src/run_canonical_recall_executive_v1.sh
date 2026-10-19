#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/continuous_brain_v0.12.2}"
cd "$PROJECT_ROOT"

ROOT="state/canonical_recall_executive_v1"
mkdir -p "$ROOT"

echo "=== v0.14.0 Canonical Recalled-Content Executive ==="
echo "Fresh v3 family: 1600 TRAIN / 400 DEV / 400 sealed TEST"
echo "Executive reads canonical recalled content, never raw memory64."

python audit_canonical_executive_family_v3.py \
  2>&1 | tee "$ROOT/family_audit.log"

grep -q 'CANONICAL_EXECUTIVE_FAMILY_AUDIT_V3: PASS' \
  "$ROOT/family_audit.log"

python qualify_canonical_recall_executive_v1.py \
  2>&1 | tee "$ROOT/qualification.log"

if grep -Eqi \
  'fall back to run on the CPU|falling back to CPU|CPU fallback' \
  "$ROOT/qualification.log"
then
  echo "CANONICAL_RECALL_EXECUTIVE_RUN: FAIL_CPU_FALLBACK_QUALIFICATION"
  exit 3
fi

grep -q 'CANONICAL_RECALL_EXECUTIVE_QUALIFICATION_V1: PASS' \
  "$ROOT/qualification.log"
grep -q 'admitted_brain_bit_exact: true' "$ROOT/qualification.log"

RESUME_ARGS=()
if [ -f "$ROOT/latest.portable.pt" ]; then
  echo "resume_checkpoint=$ROOT/latest.portable.pt"
  RESUME_ARGS=(--resume "$ROOT/latest.portable.pt")
else
  echo "resume_checkpoint=none"
fi

python train_canonical_recall_executive_v1.py \
  --state-dir "$ROOT" \
  --lr 2e-4 \
  --schedule-seed 814239 \
  --init-seed 24683 \
  "${RESUME_ARGS[@]}" \
  2>&1 | tee "$ROOT/run.log"

if grep -Eqi \
  'fall back to run on the CPU|falling back to CPU|CPU fallback' \
  "$ROOT/run.log"
then
  echo "CANONICAL_RECALL_EXECUTIVE_RUN: FAIL_CPU_FALLBACK_TRAIN"
  exit 4
fi

grep -q 'CANONICAL_RECALL_EXECUTIVE_PRODUCTION_TRAINING_COMPLETE' \
  "$ROOT/run.log"
grep -q 'admitted_brain_bit_exact: true' "$ROOT/run.log"
grep -q 'historical_base_bit_exact: true' "$ROOT/run.log"
grep -q 'FRESH_DEV_TOUCHED: false' "$ROOT/run.log"
grep -q 'FRESH_TEST_TOUCHED: false' "$ROOT/run.log"

CKPT="$ROOT/latest.portable.pt"
if [ ! -f "$CKPT" ]; then
  echo "missing update-1600 canonical checkpoint"
  exit 5
fi

echo "=== Opening fresh canonical v3 DEV exactly once ==="

python evaluate_canonical_recall_executive_fresh_dev_v1.py \
  --checkpoint "$CKPT" \
  --output "$ROOT/fresh_dev_v1.json" \
  2>&1 | tee "$ROOT/fresh_dev_v1.log"

if grep -Eqi \
  'fall back to run on the CPU|falling back to CPU|CPU fallback' \
  "$ROOT/fresh_dev_v1.log"
then
  echo "CANONICAL_RECALL_EXECUTIVE_RUN: FAIL_CPU_FALLBACK_DEV"
  exit 6
fi

grep -q 'admitted_brain_bit_exact: true' "$ROOT/fresh_dev_v1.log"
grep -q 'historical_base_bit_exact: true' "$ROOT/fresh_dev_v1.log"
grep -q 'FRESH_TEST_TOUCHED: false' "$ROOT/fresh_dev_v1.log"
grep -q 'CANONICAL_RECALL_EXECUTIVE_FRESH_DEV_COMPLETE' \
  "$ROOT/fresh_dev_v1.log"

set +e
python decide_canonical_recall_executive_fresh_dev_v1.py \
  --input "$ROOT/fresh_dev_v1.json" \
  --output "$ROOT/fresh_dev_decision_v1.json" \
  2>&1 | tee "$ROOT/fresh_dev_decision_v1.log"
status=${PIPESTATUS[0]}
set -e

echo "decision_exit=$status"
echo "FRESH_TEST_REMAINS_SEALED: true"

if [ "$status" -eq 0 ]; then
  echo "CANONICAL_RECALL_EXECUTIVE_RUN: PASS"
else
  echo "CANONICAL_RECALL_EXECUTIVE_RUN: FAIL_DEV_ADMISSION"
fi

exit "$status"
