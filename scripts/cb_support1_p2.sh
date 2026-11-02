#!/usr/bin/env bash
# Run on David's PC (WSL). SUPPORT-1 PHASE 2 (lab/SUPPORT1_PROTOCOL.md §8): verify phase 0 (PHASE0.sha256: preps +
# Sol's validator PASS) and phase 1 (PHASE1.sha256: calcheck PASS, restoretest, projection) at PIN, the CODE_HASH, the
# model and the recorded GPU (both names nonempty and equal); then for seeds 20-23: train all 5 arms (+ fork if
# admitted), every fp32 evaluation (N, 5 arms x 5 sleeps x merged/unmerged, withheld U4/G4), and BOTH analyzers.
#   pod   : NVIDIA RTX 6000 Ada Generation only (phase 1's recorded card), retried every 5 min (CB_ROUNDS, default 48)
#   out   : ~/cb_results/support1/phase2/support1_phase2.tar.gz (+ sha256; copy in Downloads) and every checkpoint
#           (sha256-verified); evidence preserved + verified before any pod removal, else the pod is STOPPED and kept
#   bash cb_support1_p2.sh
PIN=6fb6c472d8a8c019d7d1984b8d09454981d5e545
REV=b968826d9c46dd6066d109eabc6255188de91218
K=$HOME/.runpod/ssh/runpodctl-ssh-key; OUT=$HOME/cb_results/support1/phase2; DL=/mnt/c/Users/davei/Downloads
D=/workspace/Continuous_brain_support1; POD=""; S=""; DONE=0; mkdir -p $OUT
preserve() {  # archive + hash every checkpoint and the results on the pod, download, verify; 0 only if verified
  $S "cd /workspace && find support1 -name '*.pt' | sort > ckpt.list && xargs -r sha256sum < ckpt.list > support1_ckpt.sha256 \
      && tar cf support1_ckpt.tar -T ckpt.list && tar czf support1_results.tar.gz --exclude='*.pt' support1 \
      && sha256sum support1_results.tar.gz > support1_results.tar.gz.sha256" \
  && $S "cd /workspace && tar cf - support1_ckpt.tar support1_ckpt.sha256 support1_results.tar.gz support1_results.tar.gz.sha256" \
     | tar xf - -C $OUT \
  && (cd $OUT && sha256sum -c --quiet support1_results.tar.gz.sha256 && rm -rf ckpt && mkdir ckpt && tar xf support1_ckpt.tar -C ckpt \
      && cd ckpt && sha256sum -c --quiet ../support1_ckpt.sha256) \
  && echo "== preserved + verified: $OUT ($(wc -l < $OUT/support1_ckpt.sha256) checkpoints)"
}
cleanup() {
  [ "$DONE" = 1 ] || [ -z "$POD" ] && return
  if [ -z "$S" ]; then echo "== cleanup: removing pod $POD (nothing ran on it)"; runpodctl pod remove $POD >/dev/null; return; fi
  $S "cat /workspace/boot.log /workspace/support1_p*.log 2>/dev/null" > $OUT/failed_pod.log 2>/dev/null
  # explicit three-way evidence check (Sol): EVIDENCE / NONE only after a successful inspection; anything else
  # (ssh failure, unreadable directory, find error, indeterminate output) -> stop and KEEP the pod
  local ev
  ev=$($S 'if [ ! -e /workspace/support1 ]; then echo NONE; exit 0; fi
           f=$(find /workspace/support1 \( -name "*.pt" -o -name "*.json" \) -print -quit) || { echo UNKNOWN; exit 0; }
           if [ -n "$f" ]; then echo EVIDENCE; else echo NONE; fi' 2>/dev/null)
  case "$ev" in
    EVIDENCE)
      if preserve; then echo "== cleanup: evidence preserved; removing pod $POD"; runpodctl pod remove $POD >/dev/null
      else echo "== cleanup: evidence could NOT be preserved - pod $POD STOPPED and KEPT (tell Claude; storage bills until removed)"
           runpodctl pod stop $POD >/dev/null; fi;;
    NONE) echo "== cleanup: confirmed no evidence on the pod; removing pod $POD"; runpodctl pod remove $POD >/dev/null;;
    *) echo "== cleanup: could not inspect the pod (${ev:-no answer}) - pod $POD STOPPED and KEPT (tell Claude)"
       runpodctl pod stop $POD >/dev/null;;
  esac
}
[[ $PIN =~ ^[0-9a-f]{40}$ ]] || { echo "STOP: launcher PIN is not a resolved commit hash"; exit 1; }
USD_PER_HR=${USD_PER_HR:-0.86}                            # L40S secure rate used for the cost projection (override if different)
trap cleanup EXIT
pid() { python3 -c 'import json,sys
try: print(json.load(open(sys.argv[1])).get("id",""))
except Exception: pass' "$1"; }
GPUS=("NVIDIA RTX 6000 Ada Generation")                 # phase 1 recorded this card; phase 2 must match
for round in $(seq ${CB_ROUNDS:-48}); do
  for GPU in "${GPUS[@]}"; do
    if runpodctl pod create --name cb-support1-p2 --template-id runpod-torch-v280 --gpu-id "$GPU" --cloud-type SECURE \
         --container-disk-in-gb 40 --volume-in-gb 80 --volume-mount-path /workspace --min-cuda-version 13.0 \
         --ssh --wait --wait-timeout 20m > ~/cb_support1_p2_pod.json 2> ~/cb_support1_p2_pod.err; then
      POD=$(pid ~/cb_support1_p2_pod.json); [ -n "$POD" ] && break 2
    fi
    LEFT=$(pid ~/cb_support1_p2_pod.err); [ -n "$LEFT" ] && { echo "== removing unreachable pod $LEFT"; runpodctl pod remove $LEFT >/dev/null; }
    grep -q 'no longer any instances' ~/cb_support1_p2_pod.err || { echo "STOP: pod create failed ($GPU):"; cat ~/cb_support1_p2_pod.err; exit 1; }
  done
  echo "== $(date +%H:%M) round $round: no NVIDIA RTX 6000 Ada Generation available (CUDA 13 hosts); retry in 5 min"; sleep 300
done
[ -n "$POD" ] || { echo "STOP: no GPU after ${CB_ROUNDS:-48} rounds"; exit 1; }
echo "== pod $POD ($GPU)"
eval "$(runpodctl ssh info $POD | python3 -c 'import sys,json;d=json.load(sys.stdin);print("IP=%s PORT=%s" % (d["ip"], d["port"]))')"
S="ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -i $K -p $PORT root@$IP"   # a dead connection fails in ~3 min
echo "== boot: run 2's exact environment, model + tokenizer, pure tests (10-15 min)"
$S "cat > /workspace/boot.sh" <<'BOOT' || { echo "STOP: could not upload boot.sh"; exit 1; }
set -Eeuo pipefail
exec > /workspace/boot.log 2>&1
trap 'rc=$?; [ $rc = 0 ] || echo "STOP: boot failed (exit $rc)"' EXIT
git clone -q --depth 300 -b being/v3-bind https://github.com/razerblade72611/Continuous_brain $D
cd $D && git checkout -q --detach $PIN && [ "$(git rev-parse HEAD)" = "$PIN" ]
python3 -m venv /workspace/venv || { python3 -m pip install -q virtualenv && python3 -m virtualenv -q /workspace/venv; }
grep -v '^Python ' results/cons0a_run2/ENV.txt | grep -v '^-e ' > /workspace/req.txt
/workspace/venv/bin/python -m pip install -q -r /workspace/req.txt pytest || { echo 'STOP: could not install run 2 environment'; exit 1; }
/workspace/venv/bin/python -c "import torch, transformers, sys; assert torch.cuda.is_available(); print('ENV python', sys.version.split()[0], 'torch', torch.__version__, 'transformers', transformers.__version__, 'gpu', torch.cuda.get_device_name(0))"
/workspace/venv/bin/python - <<PY || { echo 'STOP: download or identity check failed'; exit 1; }
import hashlib, importlib.util, json
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3-8B", revision="$REV", local_dir="/workspace/hf/Qwen3-8B", allow_patterns=["*.safetensors", "*.json", "*.txt"])
spec = importlib.util.spec_from_file_location("b", "scripts/cls0c_b.py"); b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
m = json.load(open("results/cls0b_final/seed2.json"))["model_manifest"]["files"]
assert b.manifest("/workspace/hf/Qwen3-8B") == m, "model files differ from the CLS-0B manifest"
open("/workspace/model_manifest.sha256", "w").write(hashlib.sha256(json.dumps(m, sort_keys=True).encode()).hexdigest())
print("BOOT model == CLS-0B manifest")
PY
/workspace/venv/bin/python -m pytest -q -p no:cacheprovider tests/test_support1.py tests/test_abst0.py -m "not slow" > /workspace/pytest.log 2>&1 \
  || { tail -30 /workspace/pytest.log; echo 'STOP: SUPPORT-1 tests fail in this environment'; exit 1; }
tail -1 /workspace/pytest.log
find $D -name __pycache__ -prune -exec rm -rf {} +
echo "BOOT OK"
BOOT
# boot runs detached on the pod (a dropped connection cannot stall it). Upload and start are verified; the launcher
# then polls boot.log. The 40-min deadline is checked on EVERY iteration (also after failed or empty polls); a reachable
# pod with no milestone yet is distinguished from a connection failure (POLL_OK marker). Timeout -> exit -> cleanup.
[ "$($S "test -s /workspace/boot.sh && bash -n /workspace/boot.sh && echo UPLOAD_OK" 2>/dev/null)" = UPLOAD_OK ] \
  || { echo "STOP: boot.sh did not upload intact"; exit 1; }
[ "$($S "D=$D PIN=$PIN REV=$REV nohup bash /workspace/boot.sh > /dev/null 2>&1 < /dev/null & sleep 3; test -f /workspace/boot.log && echo STARTED" 2>/dev/null)" = STARTED ] \
  || { echo "STOP: boot did not start (no boot.log)"; exit 1; }
shown=0; t0=$(date +%s); BOOT_DEADLINE=${BOOT_DEADLINE:-2400}
while :; do
  [ $(( $(date +%s) - t0 )) -gt $BOOT_DEADLINE ] && { echo "STOP: boot did not finish in $((BOOT_DEADLINE / 60)) min - see $OUT/failed_pod.log"; exit 1; }
  sleep ${BOOT_POLL:-30}
  R=$($S "grep -E '^(ENV|BOOT|STOP|[0-9]+ passed|FAILED)' /workspace/boot.log 2>/dev/null; echo __POLL_OK__" 2>/dev/null)
  case "$R" in *__POLL_OK__*) ;; *) echo "$(date +%H:%M) (pod not reachable this poll)"; continue;; esac
  L=$(printf '%s\n' "$R" | grep -v '^__POLL_OK__$')
  n=$(printf '%s\n' "$L" | grep -c .); [ "$n" -gt "$shown" ] && { printf '%s\n' "$L" | grep . | tail -n $((n - shown)); shown=$n; }
  printf '%s\n' "$L" | grep -q '^BOOT OK' && break
  printf '%s\n' "$L" | grep -q '^STOP' && { echo "STOP: boot failed - see $OUT/failed_pod.log"; exit 1; }
done
JOBF=$(mktemp); cat > $JOBF <<'JOB'
cd /workspace/Continuous_brain_support1
PY=/workspace/venv/bin/python; M=/workspace/hf/Qwen3-8B; MS=$(cat /workspace/model_manifest.sha256); R=/workspace/support1
X="$PY scripts/support1.py"; P0=results/support1/phase0; P1=results/support1/phase1; mkdir -p $R
fin() { cd /workspace && tar czf support1_phase2.tar.gz --exclude='*.pt' support1 && sha256sum support1_phase2.tar.gz > support1_phase2.tar.gz.sha256
  echo "BUNDLE $(cat support1_phase2.tar.gz.sha256)"; echo "SUPPORT1 P2 DONE exit=$1"; exit $1; }
(cd $P0 && grep -v '^CODE_HASH \|^MODEL_MANIFEST ' PHASE0.sha256 | sha256sum -c --quiet -) || { echo "STOP: phase-0 artifacts differ from PHASE0.sha256"; fin 1; }
(cd $P1 && grep -v '^PIN \|^MODEL_MANIFEST \|^CODE_HASH \|^GPU ' PHASE1.sha256 | sha256sum -c --quiet -) || { echo "STOP: phase-1 artifacts differ from PHASE1.sha256"; fin 1; }
CH=$($X codehash)
[ "$(grep '^CODE_HASH ' $P0/PHASE0.sha256 | cut -d' ' -f2)" = "$CH" ] && [ "$(grep '^CODE_HASH ' $P1/PHASE1.sha256 | cut -d' ' -f2)" = "$CH" ] \
  || { echo "STOP: CODE_HASH differs from phase 0/1"; fin 1; }
[ "$(grep '^MODEL_MANIFEST ' $P0/PHASE0.sha256 | cut -d' ' -f2)" = "$MS" ] && [ "$(grep '^MODEL_MANIFEST ' $P1/PHASE1.sha256 | cut -d' ' -f2)" = "$MS" ] \
  || { echo "STOP: model differs from phase 0/1"; fin 1; }
G1=$(grep '^GPU ' $P1/PHASE1.sha256 | cut -d' ' -f2-); G2=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
[ -n "$G1" ] && [ -n "$G2" ] && [ "$G1" = "$G2" ] || { echo "STOP: GPU '$G2' is not phase 1's recorded '$G1'"; fin 1; }
$PY -c "import json,sys; c=json.load(open('$P1/cal/calcheck.json')); sys.exit(0 if c['pass'] is True and not c['errors'] else 1)" \
  || { echo "STOP: calibration did not pass"; fin 1; }
read FORK QD < <($PY -c "import json; c=json.load(open('$P1/projection.json'))['choice']; print(int(bool(c['fork'])), c['qdiag'])") \
  || { echo "STOP: no projection choice"; fin 1; }
FF=""; [ "$FORK" = 1 ] && FF="--fork"
echo "== plan: fork=$FORK qdiag=$QD gpu=$G2 code=$CH"
for s in 20 21 22 23; do
  O=$R/seed$s; mkdir -p $O; cp $P0/prep_seed$s.json $O/; cp $P0/prep_seed$s.json $R/
  echo "== seed $s train $(date -u +%FT%TZ)"
  $X train --prep $O/prep_seed$s.json --model $M --out $O $FF --model-manifest-sha $MS --validator-report $P0/validation.json \
     --calibration-result $P1/cal/calcheck.json --calibration-prep $P0/prep_seed24.json || fin 1
  cp $O/train_seed$s.json $R/
  for L in 1 2 3 4 5; do
    $X evalck --prep $O/prep_seed$s.json --model $M --sleep $L --out $O/eval_N_s$L.json --qdiag $QD --model-manifest-sha $MS || fin 1
    for arm in P U1 G1 U4 G4; do for mode in merged unmerged; do
      $X evalck --prep $O/prep_seed$s.json --model $M --sleep $L --arm $arm --ckpt $O/cortex_${arm}_s$L.pt --mode $mode \
         --out $O/eval_${arm}_s${L}_$mode.json --qdiag $QD --model-manifest-sha $MS || fin 1
    done; done
  done
  if [ "$FORK" = 1 ]; then for arm in U4 G4; do for mode in merged unmerged; do
    $X evalck --prep $O/prep_seed$s.json --model $M --sleep 4 --arm $arm --ckpt $O/cortex_${arm}_s4_withheld.pt --mode $mode \
       --history withheld --out $O/eval_${arm}_s4_withheld_$mode.json --qdiag $QD --model-manifest-sha $MS || fin 1
  done; done; fi
done
echo "== analysis $(date -u +%FT%TZ)"
$PY scripts/support1_analyze.py --dir $R --out $R/analysis_claude.json || fin 1
$PY scripts/support1_sol_analyze.py $R --code-hash $CH --model-manifest-sha $MS --validator-report $P0/validation.json \
   --qdiag $QD $FF --gpu "$G1" --restore-report $P1/restoretest.json --phase0-dir $P0 --phase1-dir $P1 --out $R/analysis_sol.json; echo "SOL ANALYZER rc=$?"
grep -h '"verdict"' $R/analysis_claude.json $R/analysis_sol.json | head -2
fin 0
JOB
sed -i "1i USD_PER_HR=$USD_PER_HR" $JOBF        # the rate travels inside the job file (the start command is single-quoted)
# upload + start the job, each verified and retried (an unchecked ssh call silently left phase 1 not started)
JSHA=$(sha256sum < $JOBF | cut -d' ' -f1); up=0
for i in 1 2 3; do
  $S "cat > /workspace/support1_p2.sh" < $JOBF 2>/dev/null
  [ "$($S "sha256sum < /workspace/support1_p2.sh 2>/dev/null | cut -d' ' -f1" 2>/dev/null)" = "$JSHA" ] && { up=1; break; }
  echo "== job upload attempt $i not verified; retrying"; sleep ${RETRY_WAIT:-20}
done
[ $up = 1 ] || { echo "STOP: job script could not be uploaded intact"; exit 1; }
st=""
for i in 1 2 3; do       # exactly-once: mkdir is atomic; only the call that creates the marker starts the job. A retry
                         # after a lost acknowledgment sees the marker and never starts a second copy (a marker without a
                         # job log is caught by the poll loop: 3 NOLOG polls -> STOP -> cleanup).
  st=$($S 'if mkdir /workspace/support1_p2.started 2>/dev/null; then (nohup bash /workspace/support1_p2.sh > /workspace/support1_p2.log 2>&1 < /dev/null &); sleep 3; [ -e /workspace/support1_p2.log ] && echo STARTED; else echo EXISTS; fi' 2>/dev/null)
  case "$st" in STARTED|EXISTS) break;; esac
  echo "== job start attempt $i not confirmed; retrying"; sleep ${RETRY_WAIT:-20}
done
case "$st" in STARTED|EXISTS) echo "== job started";; *) echo "STOP: job did not start"; exit 1;; esac
t0=$(date +%s)
while :; do
  [ $(( $(date +%s) - t0 )) -gt ${JOB_CAP:-$((18 * 3600))} ] && { echo "== 18-hour cap reached"; exit 1; }
  sleep ${JOB_POLL:-120}
  R=$($S "if [ ! -e /workspace/support1_p2.log ]; then echo NOLOG; elif grep -q 'SUPPORT1 P2 DONE' /workspace/support1_p2.log; then echo DONE; else echo RUNNING; fi" 2>/dev/null)
  case "$R" in
    DONE) echo "== done"; break;;
    NOLOG) nolog=$(( ${nolog:-0} + 1 )); [ $nolog -ge 3 ] && { echo "STOP: the job log is missing on a reachable pod"; exit 1; };;
    RUNNING) nolog=0;;
    *) echo "$(date +%H:%M) (pod not reachable this poll)"; continue;;
  esac
  echo "$(date +%H:%M) $($S 2>/dev/null "grep -E '^(==|EVAL|TRAIN|STOP|verdict|SOL)' /workspace/support1_p2.log | tail -1 | cut -c1-150")"
done
$S "grep -vE 'Loading weights|Writing model' /workspace/support1_p2.log" | tail -40
$S "cd /workspace && tar cf - support1_phase2.tar.gz support1_phase2.tar.gz.sha256 support1_p2.log boot.log" | tar xf - -C $OUT \
  && (cd $OUT && sha256sum -c support1_phase2.tar.gz.sha256) && preserve && cp $OUT/support1_phase2.tar.gz $DL/ \
  && echo "== downloaded + verified: $OUT/support1_phase2.tar.gz (copy in Downloads); checkpoints in $OUT/ckpt" \
  && runpodctl pod remove $POD >/dev/null && echo "== pod removed" && DONE=1
[ "$DONE" = 1 ] || { echo "== download did not complete"; exit 1; }
EX=$(grep -o 'SUPPORT1 P2 DONE exit=[0-9]*' $OUT/support1_p2.log | tail -1 | cut -d= -f2); echo "== SUPPORT-1 phase 2 exit: ${EX:-UNKNOWN}"
case "$EX" in 0) exit 0;; 4) exit 4;; 5) exit 5;; *) exit 1;; esac
