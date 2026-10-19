#!/usr/bin/env bash
# CB2-S2 (r4): PPO + KL to frozen r3h-B over organ commands, executor fixed; development selection; one sealed test.
# Executes evidence/explore/cb2/s2-prereg.json.  Run from the checkout root, explore venv, PYTHONPATH=$PWD.
#   WORKERS=32 bash scripts/explore/cb2_r4_s2.sh
# Rehearsal only (never the sealed set): ITERS=2 LIVES=2 DEV0=76001 DEV1=76002 MAXSTEPS=150 REHEARSAL=76001-76002
set -euo pipefail
WORKERS=${WORKERS:-32}; ITERS=${ITERS:-50}; LIVES=${LIVES:-64}; DEV0=${DEV0:-76001}; DEV1=${DEV1:-76040}
MAXSTEPS=${MAXSTEPS:-10000}; REHEARSAL=${REHEARSAL:-}
REF=runs/explore/cb2/candidates/r3h-B; CORE=runs/explore/cb2/sleep/r2/core/core.pt
OUT=runs/explore/cb2/s2/r4; LOGS=runs/explore/cb2/logs/s2; EV=evidence/explore/cb2
mkdir -p "$(dirname "$OUT")" "$LOGS" "$EV"
python - "$REF" <<'PY'
import json, sys, hashlib, pathlib
m = json.load(open(f"{sys.argv[1]}/cb2-ind-0001/individual.json"))
w = pathlib.Path(sys.argv[1]) / "cb2-ind-0001/weights" / f"{m['weights_sha256']}.pt"
ok = (m["state_hash"] == "1b401fc81301f8d263928b4b774c403e170b84afb4bfe254830e9dfc41ac075d"
      and m["weights_sha256"] == "ab2baa674be37970d2f0016f1e10427ac3be785e9dd907e2097449ae95004ee5"
      and hashlib.sha256(w.read_bytes()).hexdigest() == m["weights_sha256"])
print("frozen r3h-B:", "OK" if ok else "MISMATCH")
sys.exit(0 if ok else 1)
PY
echo "== s2: $ITERS iterations x $LIVES lives, $WORKERS workers; development $DEV0-$DEV1"
if [ ! -f "$OUT/selection.json" ]; then
  rm -rf "$OUT"
  python scripts/explore/cb2_s2.py train --individuals "$REF" --out "$OUT" --iterations "$ITERS" --lives "$LIVES" \
    --workers "$WORKERS" --max-steps "$MAXSTEPS" --dev-seeds "$DEV0" "$DEV1" --threads 8 2>&1 | tee "$LOGS/train.log"
fi
SELF=$OUT/selection.json
SEL=$(python -c "import json;s=json.load(open('$SELF'))['selected'];print('' if s is None else s)")
if [ -z "$SEL" ] && [ -n "$REHEARSAL" ]; then          # rehearsal only: exercise install + sealed path anyway
  SEL=$ITERS; SELF=$OUT/selection-REHEARSAL.json
  python -c "import json;s=json.load(open('$OUT/selection.json'));s.update(selected=$SEL,REHEARSAL_forced=True);json.dump(s,open('$SELF','w'))"
  echo "== REHEARSAL: no eligible checkpoint; forcing iteration $SEL to exercise the rest of the pipeline"
fi
if [ -z "$SEL" ]; then
  echo "== no eligible checkpoint: by the preregistration the sealed set is not opened"
  cp "$OUT/selection.json" "$EV/s2-selection.json"; exit 0
fi
CK=$OUT/it$(printf %03d "$SEL")
echo "== selected iteration $SEL ($CK)"
C=runs/explore/cb2/candidates/s2-final
if [ ! -d "$C" ]; then
  mkdir -p "$C" && cp -r "$REF/cb2-ind-0001" "$C/"
  python scripts/explore/cb2_evaluate.py install --candidate --individuals "$C" --core "$CORE" --policy "$CK/policy.pt" \
    --note "S2 r4 selected iteration $SEL (PPO + KL to r3h-B)" | tee "$LOGS/install.log"
fi
# the installed candidate, under the unchanged evaluator, must reproduce its development life on world $DEV0
python scripts/explore/cb2_evaluate.py evaluate --individuals "$C" --label s2-eqcheck --seeds "$DEV0-$DEV0" \
  --steps "$MAXSTEPS" --archive runs/explore/cb2/s2/eqcheck_archive --out "$OUT/eqcheck.json" > "$LOGS/eqcheck.log" 2>&1
python - "$OUT" "$SEL" "$DEV0" <<'PY'
import json, sys
out, sel, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
d = [json.loads(l) for l in open(f"{out}/dev_lives.jsonl")]
ref = next(r for r in d if r["iteration"] == sel and r["seed"] == seed)
got = json.load(open(f"{out}/eqcheck.json"))["episodes"]["individual"][0]
ok = (ref["steps"], ref["n_unlocked"]) == (got["steps"], got["n_unlocked"])
print("installed == selected (world", seed, "):", "OK" if ok else f"MISMATCH dev={ref} eval={got['steps'], got['n_unlocked']}")
sys.exit(0 if ok else 1)
PY
cp "$OUT/selection.json" "$EV/s2-selection.json"; cp "$OUT/dev.json" "$EV/s2-dev.json"; cp "$OUT/metrics.jsonl" "$EV/s2-metrics.jsonl"
if [ -n "$REHEARSAL" ]; then
  ARGS=(--candidate "$C" --comparator "$REF" --selection "$SELF" --work runs/explore/cb2/s2/sealed-REHEARSAL
        --rehearsal-seeds "$REHEARSAL" --out "$EV/s2-sealed-REHEARSAL.json")
else
  ARGS=(--candidate "$C" --comparator "$REF" --selection "$SELF" --work runs/explore/cb2/s2/sealed)
fi
python scripts/explore/cb2_sealed_s2.py "${ARGS[@]}" 2>&1 | tee "$LOGS/sealed.log"
echo "== s2 finished"
