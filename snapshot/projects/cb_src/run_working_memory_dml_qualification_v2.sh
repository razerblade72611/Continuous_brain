#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/continuous_brain_v0.12.2}"
cd "$PROJECT_ROOT"

mkdir -p state/working_memory_m1
LOG="state/working_memory_m1/dml_qualification_v2.log"

set +e
python qualify_working_memory_dml_v2.py 2>&1 | tee "$LOG"
status=${PIPESTATUS[0]}
set -e

echo "qualification_exit=$status"
if [ "$status" -ne 0 ]; then
  echo "WORKING_MEMORY_QUALIFICATION_RUN: FAIL"
  exit "$status"
fi

if grep -Eqi \
  'fall back to run on the CPU|falling back to CPU|CPU fallback' \
  "$LOG"
then
  echo "WORKING_MEMORY_QUALIFICATION_RUN: FAIL_CPU_FALLBACK"
  exit 3
fi

grep -q 'WORKING_MEMORY_DML_QUALIFICATION_OK' "$LOG"

echo "NO_REPORTED_CPU_FALLBACK"
echo "WORKING_MEMORY_QUALIFICATION_RUN: PASS"
