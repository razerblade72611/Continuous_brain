#!/usr/bin/env bash
# CB2 r3-H arm B: learned selector -> scripted executor organ (exploration lane, ENGINEERING_ONLY).
#
#   bash scripts/explore/cb2_r3_hier.sh            (checkout root, explore venv, PYTHONPATH=$PWD)
#
# The learned level (r2 core frozen; selector warm-started from r2 P-A's trunk) chooses an organ command every step
# from legal inputs only (core state, body sense, its memory organ's readout).  The scripted executor organ
# (continuous_brain_cb2/organ.py) carries the command out and never chooses the goal; every delegated action is
# logged (--telemetry).  Labels: the organ command serving the routine's intent (organ.command_for) on practice
# run 4.  Arm A (routine intent -> organ) is re-run here as the same-code reference (CPU, alongside training).
# Development gate (Sol, DAV-96): B >= 0.90x the routine on BOTH score and survival on 75001-75040 before the
# sealed 90001-90040 set is spent.  Every step skips work whose output exists, so re-running resumes.
set -euo pipefail
WORKERS=${WORKERS:-24}; STEPS=${STEPS:-1000000}; RUN=${RUN:-4}; UPDATES=${UPDATES:-8000}
DEV=${DEV_SEEDS:-75001-75040}
CORE=runs/explore/cb2/sleep/r2/core/core.pt
INIT=runs/explore/cb2/sleep/r2/policy-P-A/policy.pt
OUT=runs/explore/cb2/sleep/r3h; LOGS=runs/explore/cb2/logs/r3h; EV=evidence/explore/cb2
CANON=runs/explore/cb2/individuals/cb2-ind-0001
R=$(printf %03d "$RUN")
mkdir -p "$(dirname "$OUT")" "$LOGS" "$EV"
for f in "$CORE" "$INIT"; do
  [ "$(cut -d' ' -f1 "$f.sha256")" = "$(sha256sum "$f" | cut -d' ' -f1)" ] || { echo "STOP: $f does not match its hash"; exit 1; }
done
echo "== r3h: practice run $RUN ($STEPS steps) with organ-command labels; selector $UPDATES updates; dev $DEV"

if [ ! -f "$EV/intent-practice-r$R.json" ] || ! ls runs/explore/cb2/intent_sidecar/practice-r$R-w00.cmd.npy >/dev/null 2>&1; then
  rm -f "$EV/intent-practice-r$R.json"
  python scripts/explore/cb2_intent.py practice --workers "$WORKERS" --steps "$STEPS" --run "$RUN" \
    2>&1 | tee "$LOGS/practice.log" | grep -v '^{"worker"' || true
  [ -f "$EV/intent-practice-r$R.json" ] || { echo "STOP: practice failed (see $LOGS/practice.log)"; exit 1; }
fi
python - "$EV/intent-practice-r$R.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
print(json.dumps({k: r[k] for k in ("steps", "wall_s", "intent_action_mismatch", "mismatch_rate")}))
sys.exit(1 if r["mismatch_rate"] > 0.01 else 0)
PY

APID=""
if [ ! -f "$EV/r3h-armA-pod.json" ]; then               # arm A on CPU while the selector trains on the GPU
  python scripts/explore/cb2_organ_check.py --seeds "$DEV" --workers 8 --log-dir "$LOGS/armA" \
    --reference "$EV/eval-r3t-full.json" --out "$EV/r3h-armA-pod.json" > "$LOGS/armA.log" 2>&1 &
  APID=$!
fi

if [ ! -f "$OUT/selector/policy.pt" ]; then
  rm -rf "$OUT/selector"
  python scripts/explore/cb2_intent.py selector --core "$CORE" --init-policy "$INIT" --out "$OUT/selector" \
    --prefix "practice-r$R-" --updates "$UPDATES" --preload 2>&1 | tee "$LOGS/selector.log"
fi

C=runs/explore/cb2/candidates/r3h-B
if [ ! -d "$C" ]; then
  mkdir -p "$C" && cp -r "$CANON" "$C/"
  python scripts/explore/cb2_evaluate.py install --candidate --individuals "$C" --core "$CORE" \
    --policy "$OUT/selector/policy.pt" --note "r3h arm B: learned selector -> scripted executor organ" | tee "$LOGS/install-B.log"
fi
if [ ! -f "$EV/eval-r3h-B.json" ]; then
  python scripts/explore/cb2_evaluate.py evaluate --individuals "$C" --label r3h-B --seeds "$DEV" \
    --telemetry "$LOGS/delegation" > "$LOGS/eval-B.log" 2>&1
fi
[ -n "$APID" ] && { wait "$APID" || echo "WARNING: arm A re-run failed (see $LOGS/armA.log); the CPU result stands"; }

python - "$EV" "$LOGS/delegation" <<'PY'
import collections, glob, json, sys
ev, tel = sys.argv[1], sys.argv[2]
e = json.load(open(f"{ev}/eval-r3h-B.json"))
s, rt = e["summary"]["individual"], e["summary"]["routine"]
ratio = {m: round(s[m] / rt[m], 3) for m in ("crafter_score", "survival_steps", "achievements_per_episode")}
cmds, status = collections.Counter(), collections.Counter()
for f in glob.glob(f"{tel}/delegation-*.jsonl"):
    for line in open(f):
        r = json.loads(line)
        cmds[r["command"]] += 1
        status[r["status"]] += 1
out = {"format": "cb2-r3h-summary/1", "status": "ENGINEERING_ONLY", "seeds": "75001-75040 (development)",
       "arm_B": {"crafter_score": s["crafter_score"], "survival_steps": s["survival_steps"],
                 "achievements_per_episode": s["achievements_per_episode"],
                 "obligations": s["obligations_on_time"]["rate"], "paired_vs_routine": e["paired_difference"]},
       "routine": {"crafter_score": rt["crafter_score"], "survival_steps": rt["survival_steps"],
                   "achievements_per_episode": rt["achievements_per_episode"], "obligations": rt["obligations_on_time"]["rate"]},
       "ratio_to_routine": ratio,
       "dev_gate_0.90_both": bool(ratio["crafter_score"] >= 0.9 and ratio["survival_steps"] >= 0.9),
       "delegated_commands": dict(cmds), "organ_status": dict(status)}
for name in ("r3h-armA-pod", "r3h-armA"):
    try:
        out[name] = json.load(open(f"{ev}/{name}.json")).get("ratio_to_routine")
    except FileNotFoundError:
        pass
open(f"{ev}/r3h-summary.json", "w").write(json.dumps(out, indent=1) + "\n")
print(json.dumps(out, indent=1))
PY
echo "== r3h finished"
