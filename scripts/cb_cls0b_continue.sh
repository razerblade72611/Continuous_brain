#!/usr/bin/env bash
# Run on David's PC (WSL). CLS-0B continuation (Sol-approved, operations only): seeds 3, 4, 5 run SEQUENTIALLY, one
# worker at a time, with the UNCHANGED scientific code at the exact pilot execution commit (so all four seeds share one
# commit identity), then the unchanged frozen analyzer on seeds 2-5. Seed 2 is the pilot's completed result already on
# the volume (verified against the committed checksums). One global cap (8 h) covers the whole continuation.
#   bash cb_cls0b_continue.sh [new|resume] [gpu="NVIDIA A40"]
# resume: continues an interrupted continuation - an unfinished seed keeps its progress JSON and bank, so its recorded
# (failed / censored) write outcomes stay terminal and stored episodes are not rewritten.
MODE=${1:-new}; GPU=${2:-NVIDIA A40}; VOL=29qhoqghqf; DC=CA-MTL-1; K=$HOME/.runpod/ssh/runpodctl-ssh-key
PIN=203653c0a8aa56a747966809455d4dddcae81daa                   # commit recorded by the pilot's seed-2 result
SEED2_JSON=294c9f322ef4b043085b5f8201998aeebee37823b7cc097372b78b9c73ddfe51
SEED2_BANK=3e46b85db72916a256f0a68d16c58bc9df648e66225a2690c381f07c28ebe9ca
D=/workspace/Continuous_brain_bind; CAP=$((8 * 3600)); t0=$(date +%s)
runpodctl network-volume get $VOL >/dev/null || { echo "STOP: volume $VOL not found"; exit 1; }
runpodctl pod create --name cb-cls0b-cont --template-id runpod-torch-v280 --gpu-id "$GPU" --data-center-ids $DC \
  --network-volume-id $VOL --volume-mount-path /workspace --ssh --wait --wait-timeout 20m > ~/cb_bind_pod.json \
  || { echo "STOP: pod create failed"; cat ~/cb_bind_pod.json; exit 1; }
POD=$(python3 -c 'import json;print(json.load(open("'"$HOME"'/cb_bind_pod.json"))["id"])') || exit 1
DONE=0                                                       # any unsuccessful exit after this point stops the pod
trap 'if [ "$DONE" != 1 ] && [ -n "$POD" ]; then echo "== cleanup: stopping pod $POD (files stay on the volume)"; runpodctl pod stop $POD; fi' EXIT
echo "== pod $POD ($GPU, $DC)"
addr() { eval "$(runpodctl ssh info $POD 2>/dev/null | python3 -c 'import sys,json
try: d=json.load(sys.stdin); print("IP=%s PORT=%s" % (d["ip"], d["port"]))
except Exception: pass')"; }
addr; S="ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 -i $K -p $PORT root@$IP"
DEADLINE=$(( t0 + CAP - 1800 ))                              # workers stop 30 min before the cap; bundling still fits
$S "MODE=$MODE VOL=$VOL D=$D PIN=$PIN SEED2_JSON=$SEED2_JSON SEED2_BANK=$SEED2_BANK DEADLINE=$DEADLINE bash -s" <<'REMOTE' || { echo "STOP: setup/launch failed - removing pod"; runpodctl pod remove $POD && POD=""; exit 1; }
set -Eeuo pipefail
findmnt -no SOURCE /workspace | grep -q networkvolumes/$VOL
cd $D
if pgrep -f '[s]cripts/cls0b.py' >/dev/null; then echo 'STOP: a CLS-0B worker is running in this pod'; exit 1; fi
if ! git diff --quiet || ! git diff --cached --quiet; then echo 'STOP: tracked files modified in the pod checkout'; git status --short; exit 1; fi
if [ -e /workspace/cls0b_continue.log ]; then
  if [ "$MODE" = resume ]; then mv /workspace/cls0b_continue.log /workspace/cls0b_continue.log.$(date -u +%Y%m%dT%H%M%SZ); echo 'resuming: previous continuation log rotated'
  else echo 'STOP: a continuation already ran (cls0b_continue.log exists) - use resume mode'; tail -3 /workspace/cls0b_continue.log; exit 1; fi
fi
git fetch -q --depth 50 origin being/v3-bind
git cat-file -e $PIN^{commit}
git checkout -q --detach $PIN                                 # results/cls0b (untracked checkpoints) stays in place
[ "$(git rev-parse HEAD)" = "$PIN" ] || { echo 'STOP: HEAD is not the pinned pilot commit'; exit 1; }
echo "scientific code at: $(git log --oneline -1)"
ln -sfn /workspace/Continuous_brain/.venv .venv
ln -sfn /workspace/Continuous_brain/models models
echo "$SEED2_JSON  results/cls0b/seed2.json" | sha256sum -c --quiet
echo "$SEED2_BANK  results/cls0b/seed2_bank.pt" | sha256sum -c --quiet
.venv/bin/python -c "import json; d=json.load(open('results/cls0b/seed2.json')); assert d['status']=='complete' and d['commit']=='$PIN', 'seed 2 is not the complete pilot result'"
for m in Qwen3-8B Qwen3-Embedding-0.6B; do [ -f /workspace/hf/$m/config.json ] || { echo "STOP: model $m missing"; exit 1; }; done
cat > /workspace/cls0b_continue.sh <<'ORCH'
#!/usr/bin/env bash
# sequential CLS-0B continuation (operations only; scientific code untouched at the pinned commit)
DEADLINE=$1
cd /workspace/Continuous_brain_bind
HF=/workspace/hf
ts() { date -u +%Y%m%dT%H%M%SZ; }
status=""
for s in 3 4 5; do
  f=results/cls0b/seed$s.json
  if [ -f $f ] && .venv/bin/python -c "import json,sys; sys.exit(0 if json.load(open('$f'))['status']=='complete' else 1)"; then
    echo "seed $s already complete - skipped"; continue; fi
  [ -e results/cls0b/seed$s.log ] && mv results/cls0b/seed$s.log results/cls0b/seed$s.log.$(ts)
  st=""; [ -e $f ] && st=$(.venv/bin/python -c "import json; print(json.load(open('$f')).get('status',''))" 2>/dev/null)
  bk=""                                                       # the exact backup made for THIS worker (never searched for by mtime)
  if [ "$st" = writing ] || [ "$st" = evaluating ]; then
    echo "seed $s: resuming interrupted run (status $st) from its progress file and bank"
    bk=$f.pre_run_backup.$(ts)
    if [ -e "$bk" ] || ! cp -p "$f" "$bk" || ! cmp -s "$f" "$bk"; then   # protects the write record if the worker stops early
      echo "STOP: could not back up seed $s progress record"; status=CLS0B_FAILED; break; fi
  elif [ -e $f ]; then mv $f $f.$(ts) || { echo "STOP: could not move $f aside"; status=CLS0B_FAILED; break; }; fi   # e.g. the pilot preflight-stop record: kept, not reused
  left=$(( DEADLINE - $(date +%s) ))
  if [ $left -lt 600 ]; then status=CLS0B_TIMEOUT; echo "global deadline reached before seed $s"; break; fi
  echo "$(date +%H:%M:%S) seed $s start (time left ${left}s)"
  timeout --signal=TERM $left .venv/bin/python scripts/cls0b.py --seed $s --model $HF/Qwen3-8B \
      --external $HF/Qwen3-Embedding-0.6B --out $f > results/cls0b/seed$s.log 2>&1; e=$?
  echo "$(date +%H:%M:%S) seed $s exit $e"; tail -n 1 results/cls0b/seed$s.log
  if [ $e -eq 4 ]; then
    status=CLS0B_PREFLIGHT_STOP
    if [ -n "$bk" ]; then                                     # a preflight stop must not erase an earlier write record
      if mv "$f" "$f.preflight_stop.$(ts)" && cp -p "$bk" "$f" && cmp -s "$bk" "$f"; then
        echo "restored seed $s progress record from $bk"
      else echo "STOP: restore of seed $s progress record from $bk FAILED - backup kept"; status=CLS0B_FAILED; fi
    fi
    break; fi
  if [ $e -eq 124 ]; then status=CLS0B_TIMEOUT; break; fi
  if [ $e -ne 0 ]; then status=CLS0B_FAILED; break; fi
done
all=1; for s in 2 3 4 5; do
  .venv/bin/python -c "import json,sys; sys.exit(0 if json.load(open('results/cls0b/seed$s.json'))['status']=='complete' else 1)" 2>/dev/null || all=0
done
# operational status of every seed (never a scientific verdict)
.venv/bin/python - > results/cls0b/continuation_status.txt 2>&1 <<'PY'
import json, os
for s in (2, 3, 4, 5):
    p = f"results/cls0b/seed{s}.json"
    if not os.path.exists(p):
        print(f"seed {s}: no result"); continue
    d = json.load(open(p))
    ok = sum(1 for w in d.get("writes", {}).values() if w.get("ok"))
    print(f"seed {s}: status={d.get('status')} commit={d.get('commit','')[:7]} stored={ok}/{len(d.get('episodes', []))} "
          f"checkpoints={len(d.get('checkpoints', []))} write_s={d.get('write_seconds')} preflight={d.get('preflight')}")
PY
cat results/cls0b/continuation_status.txt
if [ $all -eq 1 ]; then
  .venv/bin/python scripts/cls0b_analyze.py results/cls0b/seed2.json results/cls0b/seed3.json results/cls0b/seed4.json \
      results/cls0b/seed5.json > results/cls0b/analysis_final.txt 2>&1; ea=$?
  cat results/cls0b/analysis_final.txt
  if [ $ea -eq 0 ]; then status=CLS0B_COMPLETE; elif [ $ea -eq 3 ]; then status=CLS0B_COLLECTED_INVALID; else status=CLS0B_ANALYSIS_FAILED; fi
else
  status=${status:-CLS0B_PARTIAL}; echo "not all four seeds complete - no scientific verdict"
fi
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
tar czf /workspace/results_cls0b_final.tar.gz results/cls0b
sha256sum /workspace/results_cls0b_final.tar.gz | sed 's|/workspace/||' > /workspace/results_cls0b_final.tar.gz.sha256 && sync
echo "BUNDLE $(cat /workspace/results_cls0b_final.tar.gz.sha256)"
echo "CLS-0B CONTINUATION DONE status=$status"
ORCH
nohup bash /workspace/cls0b_continue.sh $DEADLINE > /workspace/cls0b_continue.log 2>&1 < /dev/null &
sleep 5; echo "CONTINUATION LAUNCHED"; cat /workspace/cls0b_continue.log
REMOTE
echo "== waiting for the CLS-0B continuation (checks every 5 min, global cap 8 h)"
while :; do
  sleep 300; addr; S="ssh -o ConnectTimeout=10 -i $K -p $PORT root@$IP"
  if $S "grep -q 'CLS-0B CONTINUATION DONE' /workspace/cls0b_continue.log" 2>/dev/null; then echo "== done"; break; fi
  echo "$(date +%H:%M) running: $($S "tail -n 1 /workspace/cls0b_continue.log; for s in 3 4 5; do [ -f $D/results/cls0b/seed\$s.log ] && tail -n 1 $D/results/cls0b/seed\$s.log | cut -c1-80; done" 2>/dev/null | tr '\n' ' ')"
  [ $(( $(date +%s) - t0 )) -gt $CAP ] && { echo "== 8-hour cap reached - stopping pod (results stay on the volume)"; runpodctl pod stop $POD; POD=""; exit 1; }
done
$S "cat /workspace/cls0b_continue.log"
mkdir -p ~/cb_results/cls0b_final
$S "cd /workspace && tar cf - results_cls0b_final.tar.gz results_cls0b_final.tar.gz.sha256 cls0b_continue.log" | tar xf - -C ~/cb_results/cls0b_final \
  && (cd ~/cb_results/cls0b_final && sha256sum -c results_cls0b_final.tar.gz.sha256) \
  && cp ~/cb_results/cls0b_final/results_cls0b_final.tar.gz /mnt/c/Users/davei/Downloads/ \
  && echo "== downloaded + verified: ~/cb_results/cls0b_final (copy in Downloads)" \
  && runpodctl pod remove $POD >/dev/null && echo "== pod removed (results also stay on the volume)" && DONE=1
[ "$DONE" = 1 ] || { echo "== download/verification did not complete - pod stopped by cleanup, files stay on the volume"; exit 1; }
ST=$(grep -o 'status=CLS0B_[A-Z_]*' ~/cb_results/cls0b_final/cls0b_continue.log | tail -n 1 | cut -d= -f2)
echo "== CLS-0B status: ${ST:-UNKNOWN}"
case "$ST" in CLS0B_COMPLETE) exit 0;; CLS0B_COLLECTED_INVALID) exit 3;; CLS0B_PREFLIGHT_STOP) exit 4;; *) exit 1;; esac
