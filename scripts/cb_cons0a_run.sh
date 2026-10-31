#!/usr/bin/env bash
# Run on David's PC (WSL). CONS-0A (lab/CONS0A_PROTOCOL.md): consolidation development mechanism test.
# Creates an A40 pod on volume 29qhoqghqf (CA-MTL-1); SEPARATE checkout pinned to the frozen CONS-0A commit (other checkouts
# and their results are not touched). Job:
#   preflight - pinned HEAD; model files == CLS-0B manifests; CLS-0B banks/inputs == FILES.sha256; CLS-0B stream tokenizer
#               present (prep then fails closed unless the stream design, all 52 bank texts and 52 reconstructions match);
#               same-volume lock
#   calibrate - seed 6, QA family from raw text (Amendment 2.2): chooses E; STOP (exit 4) if the base model confabulates
#               > 0.10 on near-misses under the answer contract (protocol: revise the contract once, before 0A), if no E
#               passes (Amendment 2.2: no default E) or if the projected campaign exceeds 6 h
#   per seed 2..5 - prep (reconstruct from the bank) -> qa -> run (5 arms x 5 sleeps) -> fresh merged evaluations;
#               a seed whose run + 8 fresh files exist is skipped (relaunch resumes); a failed seed restarts from scratch;
#               results from a different commit (results/cons0a/PIN) are moved aside, never resumed
#   analyze  - advancement criteria A1-A5 -> results/cons0a/analysis.json
# Then: verified download to ~/cb_results/cons0a (cortex checkpoints stay on the volume), pod removed.
# Exit: 0 complete; 4 calibration STOP (contract / projection); 1 anything else (pod stopped, files stay on the volume).
#   bash cb_cons0a_run.sh [gpu="NVIDIA A40"]        clear a stale lock: CB_UNLOCK=1 bash cb_cons0a_run.sh
# CB_REPLAY=prep (default): reconstruct each seed from its bank (prep; fails closed unless 52/52 exact - on the H100 only
#   42/52 were, so the default needs the A40). CB_REPLAY=run1: use run 1's committed A40 prep outputs
#   (results/cons0a_run1/seed*/replay.json, hashes pinned below, 52/52 exact) instead of prep - only once Sol approves it.
GPU=${1:-NVIDIA A40}; VOL=29qhoqghqf; DC=CA-MTL-1; K=$HOME/.runpod/ssh/runpodctl-ssh-key
PIN=271b39e653bcfd56f5d9d196914968f6fff1da37                                           # frozen CONS-0A commit
D=/workspace/Continuous_brain_cons0a; OUT=$HOME/cb_results/cons0a; DL=/mnt/c/Users/davei/Downloads
CAP=$((8 * 3600))
runpodctl network-volume get $VOL >/dev/null || { echo "STOP: volume $VOL not found"; exit 1; }
runpodctl pod create --name cb-cons0a --template-id runpod-torch-v280 --gpu-id "$GPU" --data-center-ids $DC \
  --network-volume-id $VOL --volume-mount-path /workspace --ssh --wait --wait-timeout 20m > ~/cb_cons0a_pod.json \
  || { echo "STOP: pod create failed"; cat ~/cb_cons0a_pod.json; exit 1; }
POD=$(python3 -c 'import json;print(json.load(open("'"$HOME"'/cb_cons0a_pod.json"))["id"])') || exit 1
DONE=0                                                       # any unsuccessful exit after this point stops the pod
trap 'if [ "$DONE" != 1 ] && [ -n "$POD" ]; then echo "== cleanup: stopping pod $POD (files stay on the volume)"; runpodctl pod stop $POD; fi' EXIT
echo "== pod $POD ($GPU, $DC)"
addr() { eval "$(runpodctl ssh info $POD 2>/dev/null | python3 -c 'import sys,json
try: d=json.load(sys.stdin); print("IP=%s PORT=%s" % (d["ip"], d["port"]))
except Exception: pass')"; }
addr; S="ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 -i $K -p $PORT root@$IP"
$S "VOL=$VOL D=$D PIN=$PIN CB_UNLOCK=${CB_UNLOCK:-0} CB_REPLAY=${CB_REPLAY:-prep} bash -s" <<'REMOTE' || { echo "STOP: preflight/launch failed - removing pod"; runpodctl pod remove $POD && POD=""; exit 1; }
set -Eeuo pipefail
findmnt -no SOURCE /workspace | grep -q networkvolumes/$VOL
if pgrep -f '[s]cripts/cons0a' >/dev/null; then echo 'STOP: a CONS-0A process is running in this pod'; exit 1; fi
[ -d $D ] || git clone -q --depth 200 -b being/v3-bind https://github.com/razerblade72611/Continuous_brain $D
cd $D
if ! git diff --quiet || ! git diff --cached --quiet; then echo 'STOP: tracked files modified in the CONS-0A checkout'; git status --short; exit 1; fi
git fetch -q --depth 200 origin being/v3-bind
git cat-file -e $PIN^{commit}
git checkout -q --detach $PIN
[ "$(git rev-parse HEAD)" = "$PIN" ] || { echo 'STOP: HEAD is not the pinned CONS-0A commit'; exit 1; }
echo "CONS-0A code at: $(git log --oneline -1)"
(cd results/cls0b_final && sha256sum -c --quiet FILES.sha256) || { echo 'STOP: CLS-0B banks/inputs do not match FILES.sha256'; exit 1; }
ln -sfn /workspace/Continuous_brain/.venv .venv
for m in Qwen3-8B; do [ -f /workspace/hf/$m/config.json ] || { echo "STOP: model $m missing"; exit 1; }; done
[ -f /workspace/Continuous_brain/models/qwen3-0.6b/tokenizer.json ] \
  || { echo 'STOP: the CLS-0B stream tokenizer (/workspace/Continuous_brain/models/qwen3-0.6b) is missing'; exit 1; }
echo 'hashing model files against the CLS-0B manifest (a few minutes)'
.venv/bin/python - <<'PY' || { echo 'STOP: model files differ from the CLS-0B manifest'; exit 1; }
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location("b", "scripts/cls0c_b.py"); b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
ok = b.manifest("/workspace/hf/Qwen3-8B") == json.load(open("results/cls0b_final/seed2.json"))["model_manifest"]["files"]
print("MODEL MANIFEST", "OK" if ok else "MISMATCH"); sys.exit(0 if ok else 1)
PY
if [ -e /workspace/cons0a.lock ]; then
  if [ "$CB_UNLOCK" = 1 ]; then echo "WARNING: clearing lock $(cat /workspace/cons0a.lock/owner 2>/dev/null) (CB_UNLOCK=1)"; rm -rf /workspace/cons0a.lock
  else echo "STOP: lock held by $(cat /workspace/cons0a.lock/owner 2>/dev/null) - another CONS-0A job may be running; if that pod is gone: CB_UNLOCK=1"; exit 1; fi
fi
echo "PREFLIGHT OK"
cat > /workspace/cons0a_job.sh <<'JOB'
cd /workspace/Continuous_brain_cons0a
LOCK=/workspace/cons0a.lock
mkdir $LOCK 2>/dev/null || { echo "STOP: lock held by $(cat $LOCK/owner 2>/dev/null)"; echo 'CONS-0A JOB DONE exit=1'; exit 1; }
echo "$(hostname) $(date -u +%FT%TZ) pid $$" > $LOCK/owner; trap 'rm -rf $LOCK' EXIT
rm -f /workspace/results_cons0a.tar.gz /workspace/results_cons0a.tar.gz.sha256
R=results/cons0a; M=/workspace/hf/Qwen3-8B; PY=.venv/bin/python
case "${CB_REPLAY:-prep}" in prep|run1) ;; *) echo "STOP: CB_REPLAY must be prep or run1"; echo 'CONS-0A JOB DONE exit=1'; exit 1;; esac
HEADC="$(git rev-parse HEAD) replay=${CB_REPLAY:-prep}"      # resume only results made by THIS commit and replay mode
if [ -d $R ] && [ "$(cat $R/PIN 2>/dev/null)" != "$HEADC" ]; then
  OLD=results/cons0a_stale_$(cat $R/PIN 2>/dev/null | cut -c1-7)_$(date -u +%Y%m%dT%H%M%S); mv $R $OLD
  find $OLD -name '*.pt' -delete; echo "== results from another commit moved to $OLD (checkpoints removed); not resumed"
fi
mkdir -p $R; echo "$HEADC" > $R/PIN
nvidia-smi --query-gpu=name --format=csv,noheader > $R/GPU.txt 2>/dev/null || echo unknown > $R/GPU.txt
echo "== GPU: $(cat $R/GPU.txt); replay mode: ${CB_REPLAY:-prep}"
finish() { echo "== end $(date -u +%FT%TZ)"
  tar czf /workspace/results_cons0a.tar.gz --exclude='*.pt' $R
  sha256sum /workspace/results_cons0a.tar.gz | sed 's|/workspace/||' > /workspace/results_cons0a.tar.gz.sha256 && sync
  echo "BUNDLE $(cat /workspace/results_cons0a.tar.gz.sha256)"; echo "CONS-0A JOB DONE exit=$1"; exit $1; }
echo "== calibrate (seed 6) $(date -u +%FT%TZ)"
[ -f $R/calibrate.json ] || timeout 3600 $PY scripts/cons0a.py calibrate --model $M --out $R/calibrate.json || finish 1
read E REV CSTOP PROJ < <($PY -c "import json;c=json.load(open('$R/calibrate.json'));print(c['chosen_E'],int(c['contract_revision_needed']),int(c.get('calibration_stop',True)),c['projection_s'])")
echo "calibration: E=$E contract_revision_needed=$REV calibration_stop=$CSTOP projection=${PROJ}s"
[ "$REV" = 0 ] || { echo "STOP: the base model confabulates > 0.10 on near-misses under the contract - revise it once (protocol)"; finish 4; }
[ "$CSTOP" = 0 ] || { echo "STOP: no E in {4,8,16} reached QA template acquisition >= 0.80 within the competence band (Amendment 2.2: no default)"; finish 4; }
[ "$PROJ" -le 21600 ] || { echo "STOP: projected campaign ${PROJ}s exceeds 6 h"; finish 4; }
for s in 2 3 4 5; do
  O=$R/seed$s; mkdir -p $O
  if [ -f $O/run_seed$s.json ] && [ $(ls $O/fresh_seed${s}_*_s*.json 2>/dev/null | wc -l) = 8 ]; then echo "== seed $s complete - skipped"; continue; fi
  rm -rf $O; mkdir -p $O                                     # an incomplete seed restarts from scratch (protocol)
  echo "== seed $s $(date -u +%FT%TZ)"
  if [ "${CB_REPLAY:-prep}" = run1 ]; then                   # run 1's A40 reconstructions from the store (pinned)
    cp results/cons0a_run1/seed$s/replay.json $O/replay.json
    case $s in 2) H=d7c02234a3599c9432aed5faf9565a205ca6be9b7a5acc9243898c997ea330df;;
               3) H=039a6efe92e62deb1f5cb764ea1e5c08b189e4b4ec3f889c153548f975e8bb6a;;
               4) H=c970b25d6c46c184c3e738fe41bc4d472265aa81447dc043f3c077036c99dd3f;;
               5) H=43e851579cac3a8465c4ad0081a53a07a56d1bcc63f47a89a6f51d4a54c8373f;; esac
    echo "$H  $O/replay.json" | sha256sum -c --quiet || { echo "STOP: run-1 replay for seed $s does not match its pinned hash"; finish 1; }
    $PY -c "import json,sys;r=json.load(open('$O/replay.json'));sys.exit(0 if r['exact']==r['n']==52 and r['errors']==[] and r['seed']==$s else 1)" \
      || { echo "STOP: run-1 replay for seed $s is not 52/52 exact"; finish 1; }
    echo "PREP seed $s: run-1 A40 replay used (sha256 $H, 52/52 exact)"
  else
    timeout 1800 $PY scripts/cons0a.py prep --seed $s --bank results/cls0b_final/seed${s}_bank.pt --model $M --out $O/replay.json || finish 1
  fi
  timeout 1800 $PY scripts/cons0a.py qa --replay $O/replay.json --model $M --out $O/qa.json || finish 1
  timeout 10800 $PY scripts/cons0a.py run --replay $O/replay.json --qa $O/qa.json --model $M --passes $E --out $O || finish 1
  for fam in V QA; do for kind in S I; do
    timeout 1800 $PY scripts/cons0a.py fresh --replay $O/replay.json --model $M --lora $O/cortex_${fam}_shared_s1.pt --sleep 1 --out $O/fresh_seed${s}_${kind}-${fam}_s1.json || finish 1
    timeout 1800 $PY scripts/cons0a.py fresh --replay $O/replay.json --model $M --lora $O/cortex_${kind}-${fam}_s5.pt --sleep 5 --out $O/fresh_seed${s}_${kind}-${fam}_s5.json || finish 1
  done; done
done
$PY scripts/cons0a.py analyze $R/seed2 $R/seed3 $R/seed4 $R/seed5 --out $R/analysis.json || finish 1
finish 0
JOB
CB_REPLAY=$CB_REPLAY nohup bash /workspace/cons0a_job.sh > /workspace/cons0a.log 2>&1 < /dev/null &
sleep 5; echo "CONS-0A JOB LAUNCHED"
REMOTE
t0=$(date +%s)                                               # cap counts from the job launch
echo "== waiting for CONS-0A (checks every 2 min, cap 8 h from launch)"
while :; do
  sleep 120; addr; S="ssh -o ConnectTimeout=10 -i $K -p $PORT root@$IP"
  if $S "grep -q 'CONS-0A JOB DONE' /workspace/cons0a.log" 2>/dev/null; then echo "== done"; break; fi
  echo "$(date +%H:%M) running: $($S "tail -n 1 /workspace/cons0a.log | cut -c1-110" 2>/dev/null)"
  [ $(( $(date +%s) - t0 )) -gt $CAP ] && { echo "== 8-hour cap reached - stopping pod (files stay on the volume)"; runpodctl pod stop $POD; POD=""; exit 1; }
done
$S "grep -v 'Loading weights\|generation flags\|attention mask' /workspace/cons0a.log"
mkdir -p $OUT
$S "cd /workspace && tar cf - results_cons0a.tar.gz results_cons0a.tar.gz.sha256 cons0a.log" | tar xf - -C $OUT \
  && (cd $OUT && sha256sum -c results_cons0a.tar.gz.sha256) \
  && cp $OUT/results_cons0a.tar.gz $DL/ \
  && echo "== downloaded + verified: $OUT (copy in Downloads)" \
  && runpodctl pod remove $POD >/dev/null && echo "== pod removed" && DONE=1
[ "$DONE" = 1 ] || { echo "== download/verification did not complete - pod stopped by cleanup, files stay on the volume"; exit 1; }
E=$(grep -o 'CONS-0A JOB DONE exit=[0-9]*' $OUT/cons0a.log | tail -n 1 | cut -d= -f2)
echo "== CONS-0A exit: ${E:-UNKNOWN}"
case "$E" in 0) exit 0;; 4) exit 4;; *) exit 1;; esac
