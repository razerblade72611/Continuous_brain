#!/usr/bin/env bash
# CB2 r3-D: policy-only DAgger diagnostic on the frozen r2 core (exploration lane, ENGINEERING_ONLY).
#
#   bash scripts/explore/cb2_r3_dagger.sh            (checkout root, explore venv, PYTHONPATH=$PWD)
#
# Question: how much of r2's closed-loop deficit (score 0.43x, survival 0.45x the routine on 75001-75040) is
# compounding imitation error?  Arms, both warm-started from r2's P-A policy with the r2 core frozen:
#   dagger   ITERS rounds: install the current policy in a candidate copy, collect LIVES on-policy lives on fresh
#            worlds (89,000,000 + i*100,000 + n) with the routine's counterfactual labels, train UPDATES with FRAC of
#            every batch from all DAgger data so far.
#   control  the same warm start and the same total updates (ITERS*UPDATES) on the base data only (FRAC 0).
# Both are evaluated on the development worlds 75001-75040 (inspected in r2, never trained on here).  Every step
# skips work whose output exists, so re-running resumes.  The canonical individual is only ever copied.
set -euo pipefail
ITERS=${ITERS:-3}; LIVES=${LIVES:-64}; UPDATES=${UPDATES:-4000}; FRAC=${FRAC:-0.5}; WORKERS=${WORKERS:-24}
DEV=${DEV_SEEDS:-75001-75040}
CORE=runs/explore/cb2/sleep/r2/core/core.pt
INIT=runs/explore/cb2/sleep/r2/policy-P-A/policy.pt
OUT=runs/explore/cb2/sleep/r3d; LOGS=runs/explore/cb2/logs/r3d; EV=evidence/explore/cb2
CANON=runs/explore/cb2/individuals/cb2-ind-0001
mkdir -p "$OUT" "$LOGS" "$EV"
for f in "$CORE" "$INIT"; do
  [ "$(cut -d' ' -f1 "$f.sha256")" = "$(sha256sum "$f" | cut -d' ' -f1)" ] || { echo "STOP: $f does not match its hash"; exit 1; }
done
echo "== r3d: $ITERS DAgger rounds x $LIVES lives, $UPDATES updates each, frac $FRAC; control $((ITERS*UPDATES)) updates"

candidate() {   # candidate <name> <policy>: a fresh copy of the individual with that policy installed
  local C=runs/explore/cb2/candidates/$1
  if [ ! -d "$C" ]; then
    mkdir -p "$C" && cp -r "$CANON" "$C/"
    python scripts/explore/cb2_evaluate.py install --candidate --individuals "$C" --core "$CORE" --policy "$2" \
      --note "r3d $1" | tee "$LOGS/install-$1.log"
  fi
}

CPID=""
if [ ! -f "$OUT/control/policy.pt" ]; then           # the control trains on the same GPU while DAgger collects
  rm -rf "$OUT/control"
  python scripts/explore/cb2_dagger.py train --core "$CORE" --init-policy "$INIT" --frac 0 \
    --updates $((ITERS*UPDATES)) --out "$OUT/control" --preload > "$LOGS/control.log" 2>&1 &
  CPID=$!
fi

POL=$INIT
for i in $(seq 1 "$ITERS"); do
  candidate "r3d-collect-i$i" "$POL"
  S0=$((89000000 + i*100000))
  [ -f "$EV/dagger-collect-dagger-r3d-i$i.json" ] || \
    python scripts/explore/cb2_dagger.py collect --individuals "runs/explore/cb2/candidates/r3d-collect-i$i" \
      --stream "dagger-r3d-i$i" --seeds "$S0-$((S0+LIVES-1))" --workers "$WORKERS" | tee "$LOGS/collect-i$i.log"
  if [ ! -f "$OUT/policy-i$i/policy.pt" ]; then
    rm -rf "$OUT/policy-i$i"
    python scripts/explore/cb2_dagger.py train --core "$CORE" --init-policy "$POL" --frac "$FRAC" \
      --updates "$UPDATES" --out "$OUT/policy-i$i" --preload | tee "$LOGS/train-i$i.log"
  fi
  POL=$OUT/policy-i$i/policy.pt
done
[ -n "$CPID" ] && { wait "$CPID" || { echo "STOP: control training failed (see $LOGS/control.log)"; exit 1; }; }

candidate r3d-dagger "$OUT/policy-i$ITERS/policy.pt"
candidate r3d-control "$OUT/control/policy.pt"
PIDS=""
for arm in dagger control; do                      # both evaluations at once: CPU work, one life at a time each
  if [ ! -f "$EV/eval-r3d-$arm.json" ]; then
    python scripts/explore/cb2_evaluate.py evaluate --individuals "runs/explore/cb2/candidates/r3d-$arm" \
      --label "r3d-$arm" --seeds "$DEV" > "$LOGS/eval-$arm.log" 2>&1 &
    PIDS="$PIDS $!"
  fi
done
for p in $PIDS; do wait "$p" || { echo "STOP: an evaluation failed (see $LOGS/eval-*.log)"; exit 1; }; done

python - "$EV" <<'PY'
import json, sys
ev = sys.argv[1]
r2 = {"crafter_score": 10.729, "survival_steps": 226.275, "achievements_per_episode": 7.225, "obligations": 4 / 56}
rows = {"r2 (no DAgger, 15k updates)": r2}
rt = None
for arm in ("control", "dagger"):
    e = json.load(open(f"{ev}/eval-r3d-{arm}.json"))
    s, rt = e["summary"]["individual"], e["summary"]["routine"]
    rows[f"r3d-{arm}"] = {"crafter_score": s["crafter_score"], "survival_steps": s["survival_steps"],
                          "achievements_per_episode": s["achievements_per_episode"],
                          "obligations": s["obligations_on_time"]["rate"], "paired": e["paired_difference"]}
out = {"routine": {k: rt[k] for k in ("crafter_score", "survival_steps", "achievements_per_episode")} | {"obligations": rt["obligations_on_time"]["rate"]},
       "arms": rows,
       "ratio_to_routine": {k: {m: round(v[m] / rt[m], 3) for m in ("crafter_score", "survival_steps")} for k, v in rows.items()},
       "deficit_recovered_by_dagger_vs_control": {
           m: round((rows["r3d-dagger"][m] - rows["r3d-control"][m]) / max(rt[m] - rows["r3d-control"][m], 1e-9), 3)
           for m in ("crafter_score", "survival_steps")}}
open(f"{ev}/r3d-summary.json", "w").write(json.dumps(out, indent=1) + "\n")
print(json.dumps(out, indent=1))
PY
echo "== r3d finished"
