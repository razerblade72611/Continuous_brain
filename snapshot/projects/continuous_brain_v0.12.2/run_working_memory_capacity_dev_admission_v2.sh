#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/continuous_brain_v0.12.2}"
cd "$PROJECT_ROOT"

ROOT="state/working_memory_recall_capacity_control_phase1_v1"
CKPT="$ROOT/latest.portable.pt"
OUT="$ROOT/dev_admission_v2"
mkdir -p "$OUT"

echo "=== SEALED DEV ADMISSION: winning capacity-ungated memory ==="
echo "checkpoint=$CKPT"
echo "DEV=1600..1979"
echo "TEST=1980..2359 REMAINS SEALED"

python evaluate_working_memory_capacity_dev_causal_v2.py \
  --checkpoint "$CKPT" \
  --output "$OUT/causal.json" \
  2>&1 | tee "$OUT/causal.log"

if grep -Eqi 'fall back to run on the CPU|falling back to CPU|CPU fallback' "$OUT/causal.log"; then
  echo "DEV_ADMISSION_RUN: FAIL_BRAIN_CPU_FALLBACK_CAUSAL"; exit 3
fi

python evaluate_working_memory_capacity_dev_representation_v1.py \
  --checkpoint "$CKPT" \
  --output "$OUT/representation.json" \
  2>&1 | tee "$OUT/representation.log"

if grep -Eqi 'fall back to run on the CPU|falling back to CPU|CPU fallback' "$OUT/representation.log"; then
  echo "DEV_ADMISSION_RUN: FAIL_BRAIN_CPU_FALLBACK_REPRESENTATION"; exit 3
fi

set +e
python decide_working_memory_capacity_dev_admission_v2.py \
  --causal "$OUT/causal.json" \
  --representation "$OUT/representation.json" \
  --output "$OUT/decision.json" \
  2>&1 | tee "$OUT/decision.log"
status=${PIPESTATUS[0]}
set -e

echo "decision_exit=$status"
echo "TEST_REMAINS_SEALED: true"
if [ "$status" -eq 0 ]; then
  echo "WORKING_MEMORY_CAPACITY_DEV_ADMISSION_RUN: PASS"
else
  echo "WORKING_MEMORY_CAPACITY_DEV_ADMISSION_RUN: FAIL"
fi
exit "$status"
