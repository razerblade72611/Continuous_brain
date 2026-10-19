#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/continuous_brain_v0.12.2}"
cd "$PROJECT_ROOT"

ROOT="state/option_timing_observability_v1"
CKPT="state/option_intent_executive_v1/latest.portable.pt"
mkdir -p "$ROOT"

echo "=== v0.18.1 Option Timing Observability Diagnostic ==="
echo "No training of the Continuous Brain."
echo "v3 TRAIN only; v3 DEV/TEST remain untouched."

python probe_option_timing_observability_v1.py \
  --checkpoint "$CKPT" \
  --output "$ROOT/probe_v1.json" \
  2>&1 | tee "$ROOT/probe_v1.log"

grep -q 'OPTION_TIMING_OBSERVABILITY_PROBE_COMPLETE' \
  "$ROOT/probe_v1.log"
grep -q 'admitted_brain_bit_exact: true' \
  "$ROOT/probe_v1.log"
grep -q 'V3_DEV_TOUCHED: false' "$ROOT/probe_v1.log"
grep -q 'V3_TEST_TOUCHED: false' "$ROOT/probe_v1.log"

python decide_option_timing_observability_v1.py \
  --input "$ROOT/probe_v1.json" \
  --output "$ROOT/decision_v1.json" \
  2>&1 | tee "$ROOT/decision_v1.log"

grep -q 'OPTION_TIMING_OBSERVABILITY_DECISION_COMPLETE' \
  "$ROOT/decision_v1.log"

echo "NO_CONTINUOUS_BRAIN_TRAINING_PERFORMED: true"
echo "V3_DEV_REMAINS_UNUSED: true"
echo "V3_TEST_REMAINS_SEALED: true"
echo "OPTION_TIMING_OBSERVABILITY_RUN: PASS"
