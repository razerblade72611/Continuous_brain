#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/continuous_brain_v0.12.2}"
cd "$PROJECT_ROOT"

ROOT="state/canonical_recall_executive_v1"
CKPT="$ROOT/latest.portable.pt"

if [ ! -f "$CKPT" ]; then
  echo "missing canonical update-1600 checkpoint: $CKPT"
  exit 2
fi

if [ ! -f "$ROOT/fresh_dev_v1.json" ]; then
  echo "missing already-opened fresh DEV artifact: $ROOT/fresh_dev_v1.json"
  exit 3
fi

echo "=== Fixed-final canonical TRAIN evaluation ==="

python evaluate_canonical_recall_executive_final_train_v1.py \
  --checkpoint "$CKPT" \
  --output "$ROOT/final_train_v1.json" \
  2>&1 | tee "$ROOT/final_train_v1.log"

if grep -Eqi \
  'fall back to run on the CPU|falling back to CPU|CPU fallback' \
  "$ROOT/final_train_v1.log"
then
  echo "CANONICAL_FAILURE_DIAGNOSTIC_RUN: FAIL_CPU_FALLBACK_TRAIN"
  exit 4
fi

grep -q 'CANONICAL_RECALL_EXECUTIVE_FINAL_TRAIN_COMPLETE' \
  "$ROOT/final_train_v1.log"
grep -q 'FRESH_DEV_REOPENED: false' "$ROOT/final_train_v1.log"
grep -q 'FRESH_TEST_TOUCHED: false' "$ROOT/final_train_v1.log"

echo "=== TRAIN->DEV representation portability ==="

python probe_canonical_content_portability_v1.py \
  --checkpoint "$CKPT" \
  --output "$ROOT/content_portability_v1.json" \
  2>&1 | tee "$ROOT/content_portability_v1.log"

if grep -Eqi \
  'fall back to run on the CPU|falling back to CPU|CPU fallback' \
  "$ROOT/content_portability_v1.log"
then
  echo "CANONICAL_FAILURE_DIAGNOSTIC_RUN: FAIL_BRAIN_CPU_FALLBACK_PORTABILITY"
  exit 5
fi

grep -q 'CANONICAL_CONTENT_PORTABILITY_PROBE_COMPLETE' \
  "$ROOT/content_portability_v1.log"
grep -q 'FRESH_TEST_TOUCHED: false' "$ROOT/content_portability_v1.log"

echo "=== Combined diagnosis ==="

python diagnose_canonical_failure_v1.py \
  --train "$ROOT/final_train_v1.json" \
  --dev "$ROOT/fresh_dev_v1.json" \
  --portability "$ROOT/content_portability_v1.json" \
  --output "$ROOT/failure_diagnosis_v1.json" \
  2>&1 | tee "$ROOT/failure_diagnosis_v1.log"

grep -q 'CANONICAL_FAILURE_DIAGNOSIS_COMPLETE' \
  "$ROOT/failure_diagnosis_v1.log"
grep -q 'FRESH_TEST_TOUCHED: false' \
  "$ROOT/failure_diagnosis_v1.log"

echo "NO_TRAINING_PERFORMED: true"
echo "FRESH_TEST_REMAINS_SEALED: true"
echo "CANONICAL_FAILURE_DIAGNOSTIC_RUN: PASS"
