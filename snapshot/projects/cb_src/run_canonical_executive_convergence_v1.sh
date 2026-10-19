#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/continuous_brain_v0.12.2}"
cd "$PROJECT_ROOT"

ROOT="state/canonical_recall_executive_convergence_v1"
ORIGINAL="state/canonical_recall_executive_v1/latest.portable.pt"

mkdir -p "$ROOT"

if [ ! -f "$ORIGINAL" ]; then
  echo "missing original update-1600 canonical checkpoint: $ORIGINAL"
  exit 2
fi

echo "=== Canonical Executive Convergence v1 ==="
echo "Architecture changes: NONE"
echo "Starting scientific fact: v0.14.0 underlearned on TRAIN."
echo "v3 DEV will NOT be used for optimization."
echo "v3 TEST remains sealed."

# If a previous invocation already passed the convergence gate, do not train more.
if [ -f "$ROOT/convergence_decision_latest.json" ]; then
  if python - "$ROOT/convergence_decision_latest.json" <<'PY'
import json, sys
p=json.load(open(sys.argv[1]))
raise SystemExit(0 if p.get("pass") is True else 1)
PY
  then
    echo "existing_convergence_gate: PASS"
    echo "CANONICAL_EXECUTIVE_CONVERGENCE_RUN: PASS"
    exit 0
  fi
fi

get_update () {
  local ckpt="$1"
  python - "$ckpt" <<'PY'
import sys, torch
p=torch.load(sys.argv[1],map_location="cpu",weights_only=True)
print(int(p.get("global_update",p.get("update",-1))))
PY
}

evaluate_gate () {
  local target="$1"

  echo "=== Fixed-final TRAIN evaluation at update $target ==="

  python evaluate_canonical_convergence_final_train_v1.py \
    --checkpoint "$ROOT/latest.portable.pt" \
    --output "$ROOT/final_train_u${target}.json" \
    2>&1 | tee "$ROOT/final_train_u${target}.log"

  if grep -Eqi \
    'fall back to run on the CPU|falling back to CPU|CPU fallback' \
    "$ROOT/final_train_u${target}.log"
  then
    echo "CANONICAL_EXECUTIVE_CONVERGENCE_RUN: FAIL_CPU_FALLBACK_EVAL"
    exit 5
  fi

  grep -q 'CANONICAL_EXECUTIVE_CONVERGENCE_FINAL_TRAIN_COMPLETE' \
    "$ROOT/final_train_u${target}.log"
  grep -q 'FRESH_DEV_REOPENED: false' \
    "$ROOT/final_train_u${target}.log"
  grep -q 'FRESH_TEST_TOUCHED: false' \
    "$ROOT/final_train_u${target}.log"

  set +e
  python decide_canonical_executive_convergence_train_v1.py \
    --input "$ROOT/final_train_u${target}.json" \
    --output "$ROOT/convergence_decision_u${target}.json" \
    2>&1 | tee "$ROOT/convergence_decision_u${target}.log"
  local status=${PIPESTATUS[0]}
  set -e

  cp "$ROOT/convergence_decision_u${target}.json" \
    "$ROOT/convergence_decision_latest.json"

  return "$status"
}

# Stages are deliberately TRAIN-only.
# Epoch 2 retains the original LR.
# Later epochs decay it to reduce the instability seen near update 1600.
TARGETS=(3200 4800 6400 8000)
LRS=(0.0002 0.0001 0.00005 0.000025)

current=1600
if [ -f "$ROOT/latest.portable.pt" ]; then
  current="$(get_update "$ROOT/latest.portable.pt")"
fi

echo "current_global_update=$current"

for i in "${!TARGETS[@]}"; do
  target="${TARGETS[$i]}"
  lr="${LRS[$i]}"

  if [ "$current" -lt "$target" ]; then
    echo "=== Continue TRAIN optimization to update $target at lr=$lr ==="

    python train_canonical_executive_convergence_v1.py \
      --state-dir "$ROOT" \
      --original-checkpoint "$ORIGINAL" \
      --target-update "$target" \
      --lr "$lr" \
      2>&1 | tee "$ROOT/train_to_u${target}.log"

    if grep -Eqi \
      'fall back to run on the CPU|falling back to CPU|CPU fallback' \
      "$ROOT/train_to_u${target}.log"
    then
      echo "CANONICAL_EXECUTIVE_CONVERGENCE_RUN: FAIL_CPU_FALLBACK_TRAIN"
      exit 3
    fi

    grep -q 'CANONICAL_EXECUTIVE_CONVERGENCE_TRAINING_COMPLETE' \
      "$ROOT/train_to_u${target}.log"
    grep -q 'admitted_brain_bit_exact: true' \
      "$ROOT/train_to_u${target}.log"
    grep -q 'historical_base_bit_exact: true' \
      "$ROOT/train_to_u${target}.log"
    grep -q 'V3_DEV_TOUCHED: false' \
      "$ROOT/train_to_u${target}.log"
    grep -q 'V3_TEST_TOUCHED: false' \
      "$ROOT/train_to_u${target}.log"

    current="$(get_update "$ROOT/latest.portable.pt")"
  fi

  if [ "$current" -eq "$target" ]; then
    if evaluate_gate "$target"; then
      echo "CONVERGENCE_REACHED_AT_UPDATE=$target"
      echo "V3_DEV_USED_FOR_OPTIMIZATION: false"
      echo "V3_TEST_REMAINS_SEALED: true"
      echo "CANONICAL_EXECUTIVE_CONVERGENCE_RUN: PASS"
      exit 0
    fi

    echo "TRAIN convergence gate still failed at update $target"
  fi
done

echo "V3_DEV_USED_FOR_OPTIMIZATION: false"
echo "V3_TEST_REMAINS_SEALED: true"
echo "CANONICAL_EXECUTIVE_CONVERGENCE_RUN: FAIL_AFTER_MAX_EPOCHS"
exit 4
