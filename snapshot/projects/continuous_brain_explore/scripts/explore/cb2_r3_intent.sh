#!/usr/bin/env bash
# CB2 r3-T: the target-given diagnostic on the frozen r2 core (exploration lane, ENGINEERING_ONLY, DIAGNOSTIC).
#
#   bash scripts/explore/cb2_r3_intent.sh            (checkout root, explore venv, PYTHONPATH=$PWD)
#
# Question: does knowing the routine's intent (which routine, and where it is walking) lift the learned skill's
# closed-loop survival from r2's 0.45x toward the routine?  Three arms on the same batches (cb2_intent.py):
# none (control), branch, full.  All warm-started from r2 P-A, r2 core frozen, evaluated on the development worlds
# 75001-75040.  Preregistered reading (DAV-88, before any result):
#   - full or branch >= 0.80x routine survival AND >= +0.20 over none: hidden intent is the missing variable;
#     r3-H gives the learned level the intent (option + target), learned or scripted executor below it.
#   - full ~ none (within 0.10): intent is not the bottleneck; the executor (navigation from the core's state)
#     is; r3-H keeps the scripted executor organ for now and the memory track moves up.
#   - branch ~ full: the learned level only needs to choose the routine (option); full >> branch: it must also
#     choose targets.
# Every step skips work whose output exists, so re-running resumes.  The canonical individual is only copied.
set -euo pipefail
WORKERS=${WORKERS:-24}; STEPS=${STEPS:-1000000}; RUN=${RUN:-4}; UPDATES=${UPDATES:-8000}
DEV=${DEV_SEEDS:-75001-75040}; ARMS=${ARMS:-none,branch,full}
CORE=runs/explore/cb2/sleep/r2/core/core.pt
INIT=runs/explore/cb2/sleep/r2/policy-P-A/policy.pt
OUT=runs/explore/cb2/sleep/r3t; LOGS=runs/explore/cb2/logs/r3t; EV=evidence/explore/cb2
CANON=runs/explore/cb2/individuals/cb2-ind-0001
mkdir -p "$(dirname "$OUT")" "$LOGS" "$EV"
for f in "$CORE" "$INIT"; do
  [ "$(cut -d' ' -f1 "$f.sha256")" = "$(sha256sum "$f" | cut -d' ' -f1)" ] || { echo "STOP: $f does not match its hash"; exit 1; }
done
echo "== r3t: practice run $RUN ($STEPS steps, $WORKERS workers) with intent; arms $ARMS, $UPDATES updates"

if [ ! -f "$EV/intent-practice-r$(printf %03d "$RUN").json" ]; then
  python scripts/explore/cb2_intent.py practice --workers "$WORKERS" --steps "$STEPS" --run "$RUN" \
    2>&1 | tee "$LOGS/practice.log" | grep -v '^{"worker"' || true
  [ -f "$EV/intent-practice-r$(printf %03d "$RUN").json" ] || { echo "STOP: practice failed (see $LOGS/practice.log)"; exit 1; }
fi
python - "$EV/intent-practice-r$(printf %03d "$RUN").json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
print(json.dumps({k: r[k] for k in ("steps", "wall_s", "intent_action_mismatch", "mismatch_rate")}))
sys.exit(1 if r["mismatch_rate"] > 0.01 else 0)
PY

if [ ! -f "$OUT/full/policy.pt" ] && [ ! -f "$OUT/none/policy.pt" ]; then
  rm -rf "$OUT"
  python scripts/explore/cb2_intent.py train --core "$CORE" --init-policy "$INIT" --out "$OUT" --arms "$ARMS" \
    --prefix "practice-r$(printf %03d "$RUN")-" --updates "$UPDATES" --preload 2>&1 | tee "$LOGS/train.log"
fi

PIDS=""
for arm in ${ARMS//,/ }; do
  C=runs/explore/cb2/candidates/r3t-$arm
  if [ ! -d "$C" ]; then
    mkdir -p "$C" && cp -r "$CANON" "$C/"
    python scripts/explore/cb2_evaluate.py install --candidate --individuals "$C" --core "$CORE" \
      --policy "$OUT/$arm/policy.pt" --note "r3t $arm (DIAGNOSTIC: privileged intent)" | tee "$LOGS/install-$arm.log"
  fi
  if [ ! -f "$EV/eval-r3t-$arm.json" ]; then            # the evaluations run at once: CPU, one life at a time each
    python scripts/explore/cb2_evaluate.py evaluate --individuals "$C" --label "r3t-$arm" --seeds "$DEV" \
      > "$LOGS/eval-$arm.log" 2>&1 &
    PIDS="$PIDS $!"
  fi
done
for p in $PIDS; do wait "$p" || { echo "STOP: an evaluation failed (see $LOGS/eval-*.log)"; exit 1; }; done

python scripts/explore/cb2_intent.py summary --label r3t --arms "$ARMS"
echo "== r3t finished"
