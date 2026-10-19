#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/continuous_brain_v0.12.2}"
cd "$PROJECT_ROOT"

ROOT="state/executive_novel_rules_v2"
mkdir -p "$ROOT"

echo "=== v0.13.2 Fresh Novel-Rule Executive ==="
echo "Fresh family: 1600 TRAIN / 400 DEV / 400 sealed TEST"
echo "All active rules are novel relative to the admitted NORMAL policy."

python audit_executive_novel_rule_family_v2.py \
  2>&1 | tee "$ROOT/family_audit.log"
grep -q 'EXECUTIVE_NOVEL_RULE_FAMILY_AUDIT_V2: PASS' "$ROOT/family_audit.log"

python qualify_executive_novel_rules_v2.py \
  2>&1 | tee "$ROOT/qualification.log"

if grep -Eqi \
  'fall back to run on the CPU|falling back to CPU|CPU fallback' \
  "$ROOT/qualification.log"
then
  echo "EXECUTIVE_NOVEL_RULE_V2_RUN: FAIL_CPU_FALLBACK_QUALIFICATION"
  exit 3
fi
grep -q 'EXECUTIVE_NOVEL_RULE_QUALIFICATION_V2: PASS' "$ROOT/qualification.log"
grep -q 'admitted_brain_bit_exact: true' "$ROOT/qualification.log"

RESUME_ARGS=()
if [ -f "$ROOT/latest.portable.pt" ]; then
  echo "resume_checkpoint=$ROOT/latest.portable.pt"
  RESUME_ARGS=(--resume "$ROOT/latest.portable.pt")
else
  echo "resume_checkpoint=none"
fi

python train_executive_novel_rules_v2.py \
  --state-dir "$ROOT" \
  --lr 2e-4 \
  --schedule-seed 731927 \
  --init-seed 24682 \
  "${RESUME_ARGS[@]}" \
  2>&1 | tee "$ROOT/run.log"

if grep -Eqi \
  'fall back to run on the CPU|falling back to CPU|CPU fallback' \
  "$ROOT/run.log"
then
  echo "EXECUTIVE_NOVEL_RULE_V2_RUN: FAIL_CPU_FALLBACK_TRAIN"
  exit 4
fi

grep -q 'EXECUTIVE_NOVEL_RULE_PRODUCTION_TRAINING_COMPLETE' "$ROOT/run.log"
grep -q 'admitted_brain_bit_exact: true' "$ROOT/run.log"
grep -q 'historical_base_bit_exact: true' "$ROOT/run.log"
grep -q 'FRESH_DEV_TOUCHED: false' "$ROOT/run.log"
grep -q 'FRESH_TEST_TOUCHED: false' "$ROOT/run.log"

CKPT="$ROOT/latest.portable.pt"
if [ ! -f "$CKPT" ]; then
  echo "missing update-1600 checkpoint"
  exit 5
fi

echo "=== Opening fresh v2 DEV exactly once ==="

python evaluate_executive_novel_rules_fresh_dev_v2.py \
  --checkpoint "$CKPT" \
  --output "$ROOT/fresh_dev_v2.json" \
  2>&1 | tee "$ROOT/fresh_dev_v2.log"

if grep -Eqi \
  'fall back to run on the CPU|falling back to CPU|CPU fallback' \
  "$ROOT/fresh_dev_v2.log"
then
  echo "EXECUTIVE_NOVEL_RULE_V2_RUN: FAIL_CPU_FALLBACK_DEV"
  exit 6
fi

grep -q 'admitted_brain_bit_exact: true' "$ROOT/fresh_dev_v2.log"
grep -q 'historical_base_bit_exact: true' "$ROOT/fresh_dev_v2.log"
grep -q 'FRESH_TEST_TOUCHED: false' "$ROOT/fresh_dev_v2.log"
grep -q 'EXECUTIVE_NOVEL_RULE_FRESH_DEV_EVAL_COMPLETE' "$ROOT/fresh_dev_v2.log"

set +e
python decide_executive_novel_rules_fresh_dev_v2.py \
  --input "$ROOT/fresh_dev_v2.json" \
  --output "$ROOT/fresh_dev_decision_v2.json" \
  2>&1 | tee "$ROOT/fresh_dev_decision_v2.log"
status=${PIPESTATUS[0]}
set -e

echo "decision_exit=$status"
echo "FRESH_TEST_REMAINS_SEALED: true"

if [ "$status" -eq 0 ]; then
  echo "EXECUTIVE_NOVEL_RULE_V2_RUN: PASS"
else
  echo "EXECUTIVE_NOVEL_RULE_V2_RUN: FAIL_DEV_ADMISSION"
fi
exit "$status"
