#!/usr/bin/env bash
# CB2-S1, the smallest end-to-end experiment (exploration lane, ENGINEERING_ONLY).
#
# Run from the checkout root, with the exploration venv active and PYTHONPATH=$PWD:
#     bash scripts/explore/cb2_s1.sh <round> [practice_steps] [core_size] [core_updates] [policy_updates]
# e.g. bash scripts/explore/cb2_s1.sh r1 1000000 S 30000 15000
#
# What it does (each step refuses to overwrite an earlier output, so a crashed run is resumed by re-running with
# the finished steps' outputs left in place; it skips any step whose output exists):
#   1. practice: (nproc - 1) workers of the routine teacher in fresh worlds, PRACTICE x TEACHER
#   2. sleep:    the predictive core, then the P-A and P-B imitation policies on the frozen core, then the probes
#   3. a CANDIDATE copy of the individual gets the P-A skill by a consolidation event, and is tested on the 40
#      engineering-evaluation worlds against the routine; the canonical individual is not touched
# The canonical install is a separate, deliberate step (cb2_evaluate.py install) once the gates are read.
set -euo pipefail
ROUND=${1:?round label, e.g. r1}
STEPS=${2:-1000000}
SIZE=${3:-S}
CORE_UPDATES=${4:-30000}
POLICY_UPDATES=${5:-15000}
WORKERS=$(( $(nproc) > 2 ? $(nproc) - 1 : 2 ))
SLEEP=runs/explore/cb2/sleep/$ROUND
CAND=runs/explore/cb2/candidates/$ROUND
LOGS=runs/explore/cb2/logs/$ROUND
mkdir -p "$LOGS" evidence/explore/cb2
RUN_NO=$(python3 -c "import re,sys; m=re.findall(r'\d+', '$ROUND'); print(int(m[-1]) if m else 1)")

echo "== CB2-S1 $ROUND: $STEPS practice steps on $WORKERS workers, core $SIZE x $CORE_UPDATES, policies x $POLICY_UPDATES"
python - <<'EOF'
import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
                    torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
EOF

if [ ! -f "$LOGS/practice.done" ]; then
  python scripts/explore/cb2_collect.py bench --workers 1,$WORKERS --seconds 60 --out evidence/explore/cb2/bench-$ROUND.json | tee "$LOGS/bench.log"
  python scripts/explore/cb2_collect.py practice --workers "$WORKERS" --steps "$STEPS" --run "$RUN_NO" | tee "$LOGS/practice.log"
  touch "$LOGS/practice.done"
fi
[ -f "$SLEEP/core/core.pt" ] || python scripts/explore/cb2_sleep.py core --size "$SIZE" --updates "$CORE_UPDATES" --out "$SLEEP/core" | tee "$LOGS/core.log"
for ARM in P-A P-B; do
  [ -f "$SLEEP/policy-$ARM/policy.pt" ] || python scripts/explore/cb2_sleep.py policy --core "$SLEEP/core/core.pt" --arm "$ARM" --updates "$POLICY_UPDATES" --out "$SLEEP/policy-$ARM" | tee "$LOGS/policy-$ARM.log"
done
[ -f "$LOGS/dream.done" ] || { python scripts/explore/cb2_sleep.py dream --core "$SLEEP/core/core.pt" --policy "$SLEEP/policy-P-B/policy.pt" --stream "dream-$ROUND" --n 64 | tee "$LOGS/dream.log"; touch "$LOGS/dream.done"; }
[ -f "evidence/explore/cb2/probe-$ROUND.json" ] || python scripts/explore/cb2_sleep.py probe --core "$SLEEP/core/core.pt" --out "evidence/explore/cb2/probe-$ROUND.json" | tee "$LOGS/probe.log"
if [ ! -d "$CAND" ]; then
  mkdir -p "$CAND"
  cp -r runs/explore/cb2/individuals/cb2-ind-0001 "$CAND/"
  python scripts/explore/cb2_evaluate.py install --individuals "$CAND" --core "$SLEEP/core/core.pt" --policy "$SLEEP/policy-P-A/policy.pt" --note "candidate $ROUND" | tee "$LOGS/install.log"
fi
[ -f "evidence/explore/cb2/eval-$ROUND-PA.json" ] || python scripts/explore/cb2_evaluate.py evaluate --individuals "$CAND" --label "$ROUND-PA" | tee "$LOGS/eval.log"
python - "$ROUND" <<'EOF'
import json, sys
r = sys.argv[1]
ev = json.load(open(f"evidence/explore/cb2/eval-{r}-PA.json"))
s, d = ev["summary"], ev["paired_difference"]
it, rt = s["individual"], s["routine"]
agree = {}
for arm in ("P-A", "P-B"):
    rows = [json.loads(l) for l in open(f"runs/explore/cb2/sleep/{r}/policy-{arm}/metrics.jsonl")]
    agree[arm] = rows[-1]["heldout_agreement"]
gates = {"held-out agreement P-A >= 0.90": agree["P-A"] >= 0.90,
         "score >= 0.8 x routine": it["crafter_score"] >= 0.8 * rt["crafter_score"],
         "survival >= 0.8 x routine": it["survival_steps"] >= 0.8 * rt["survival_steps"]}
print(json.dumps({"agreement": agree, "individual": {k: it[k] for k in ("crafter_score", "achievements_per_episode", "survival_steps")},
                  "routine": {k: rt[k] for k in ("crafter_score", "achievements_per_episode", "survival_steps")},
                  "paired": d, "gates": gates}, indent=1))
EOF
