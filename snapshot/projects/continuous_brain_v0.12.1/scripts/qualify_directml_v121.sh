#!/usr/bin/env bash
set -euo pipefail

ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-directml}"
CHECKPOINT="${CHECKPOINT:-state/child_v113_dml_smoke/infant_training.pt}"

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Checkpoint not found: $CHECKPOINT" >&2
  exit 2
fi

run_gate() {
  echo
  echo "================================================================================"
  echo "DML QUALIFICATION: $*"
  echo "================================================================================"
  "$PYTHON" -m continuous_brain.dml_qualification_v121 "$@"
}

run_gate basic --device "$DEVICE"
for n in 1 2 5 10 20; do
  run_gate gru --device "$DEVICE" --steps "$n"
done
run_gate infant-core --device "$DEVICE" --checkpoint "$CHECKPOINT"
run_gate competitive-one-step --device "$DEVICE" --checkpoint "$CHECKPOINT"
run_gate competitive-episode --device "$DEVICE" --checkpoint "$CHECKPOINT"

echo
echo "DIRECTML_V121_QUALIFIED"
