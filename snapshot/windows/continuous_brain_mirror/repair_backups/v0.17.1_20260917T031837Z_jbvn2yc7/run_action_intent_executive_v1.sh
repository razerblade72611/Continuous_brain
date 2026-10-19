#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/continuous_brain_v0.12.2}"
cd "$PROJECT_ROOT"
ROOT="state/action_intent_executive_v1"
mkdir -p "$ROOT"

echo "=== v0.17.0 Executive Action Intent Cortex ==="
echo "Memory architecture changes: NONE"
echo "Executive control interface: frozen motor-logit residual"
echo "v3 TRAIN only; v3 DEV/TEST are not used."

python qualify_action_intent_executive_v1.py 2>&1 | tee "$ROOT/qualification.log"
grep -q 'ACTION_INTENT_EXECUTIVE_QUALIFICATION_V1: PASS' "$ROOT/qualification.log"

TARGETS=(1600 3200 4800)
LRS=(0.0002 0.0001 0.00005)

get_update () {
  python - "$1" <<'PY'
import sys,torch
p=torch.load(sys.argv[1],map_location="cpu",weights_only=True)
print(int(p.get("global_update",-1)))
PY
}

evaluate_gate () {
  local target="$1"
  python evaluate_action_intent_executive_final_train_v1.py \
    --checkpoint "$ROOT/latest.portable.pt" \
    --output "$ROOT/final_train_u${target}.json" \
    2>&1 | tee "$ROOT/final_train_u${target}.log"

  grep -q 'ACTION_INTENT_EXECUTIVE_FINAL_TRAIN_COMPLETE' \
    "$ROOT/final_train_u${target}.log"

  set +e
  python decide_action_intent_executive_train_v1.py \
    --input "$ROOT/final_train_u${target}.json" \
    --output "$ROOT/train_decision_u${target}.json" \
    2>&1 | tee "$ROOT/train_decision_u${target}.log"
  local status=${PIPESTATUS[0]}
  set -e
  return "$status"
}

current=0
if [ -f "$ROOT/latest.portable.pt" ]; then
  current="$(get_update "$ROOT/latest.portable.pt")"
fi
echo "current_global_update=$current"

for i in "${!TARGETS[@]}"; do
  target="${TARGETS[$i]}"; lr="${LRS[$i]}"
  if [ "$current" -lt "$target" ]; then
    echo "=== TRAIN action-intent executive to update $target at lr=$lr ==="
    python train_action_intent_executive_v1.py \
      --state-dir "$ROOT" \
      --target-update "$target" \
      --lr "$lr" \
      2>&1 | tee "$ROOT/train_to_u${target}.log"

    grep -q 'ACTION_INTENT_EXECUTIVE_TRAINING_COMPLETE' \
      "$ROOT/train_to_u${target}.log"
    current="$(get_update "$ROOT/latest.portable.pt")"
  fi

  if [ "$current" -eq "$target" ]; then
    echo "=== Fixed-final TRAIN gate at update $target ==="
    if evaluate_gate "$target"; then
      echo "ACTION_INTENT_CONVERGED_AT_UPDATE=$target"
      echo "READY_FOR_FRESH_V4_GENERALIZATION: true"
      echo "V3_DEV_USED_FOR_SELECTION: false"
      echo "V3_TEST_REMAINS_SEALED: true"
      echo "ACTION_INTENT_EXECUTIVE_RUN: PASS"
      exit 0
    fi
    echo "action-intent TRAIN gate failed at update $target"
  fi
done

echo "READY_FOR_FRESH_V4_GENERALIZATION: false"
echo "V3_DEV_USED_FOR_SELECTION: false"
echo "V3_TEST_REMAINS_SEALED: true"
echo "ACTION_INTENT_EXECUTIVE_RUN: FAIL_AFTER_MAX_EPOCHS"
exit 4
