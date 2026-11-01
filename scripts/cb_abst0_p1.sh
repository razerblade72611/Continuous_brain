#!/usr/bin/env bash
# Run on David's PC (WSL). ABST-0 PHASE 1 (lab/ABST0_PROTOCOL.md): calibration (seed 16) + negative/QA corpora for the
# development seeds 12-15, then STOP for Claude's blind label audit (no development training happens here).
#   pod   : A40 (run 2's GPU), any data center, retried every 5 min (CB_ROUNDS, default 48); CUDA >= 13.0 host
#   boot  : venv = run 2's exact pip freeze; Qwen3-8B at the CLS-0B revision + stream tokenizer == CLS-0B manifest;
#           tests/test_abst0.py pure tests on the pod
#   job   : prep 16 -> train P sleep 1 for E in {4,8,16} -> fp32 merged + unmerged eval of each, and N -> validated
#           calchoose (STOP exit 4 if no E passes, N's exact-UNKNOWN on FRESH < 0.90, or any validation/integrity
#           error); prep 12-15 (+ audit samples)
#   out   : ~/cb_results/abst0/phase1/abst0_phase1.tar.gz (+ sha256; copy in Downloads) and the calibration checkpoints
#           (sha256-verified); on any failure with evidence on the pod, evidence is preserved + verified before removal,
#           otherwise the pod is STOPPED and kept
#   bash cb_abst0_p1.sh
PIN=34e1f9d81fdd865db92bb1bb7a06ef1ed4ff34be
REV=b968826d9c46dd6066d109eabc6255188de91218
K=$HOME/.runpod/ssh/runpodctl-ssh-key; OUT=$HOME/cb_results/abst0/phase1; DL=/mnt/c/Users/davei/Downloads
D=/workspace/Continuous_brain_abst0; POD=""; S=""; DONE=0; mkdir -p $OUT
preserve() {  # archive + hash every checkpoint and the results on the pod, download, verify; 0 only if verified
  $S "cd /workspace && find abst0 -name '*.pt' | sort > ckpt.list && xargs -r sha256sum < ckpt.list > abst0_ckpt.sha256 \
      && tar cf abst0_ckpt.tar -T ckpt.list && tar czf abst0_results.tar.gz --exclude='*.pt' abst0 \
      && sha256sum abst0_results.tar.gz > abst0_results.tar.gz.sha256" \
  && $S "cd /workspace && tar cf - abst0_ckpt.tar abst0_ckpt.sha256 abst0_results.tar.gz abst0_results.tar.gz.sha256" \
     | tar xf - -C $OUT \
  && (cd $OUT && sha256sum -c --quiet abst0_results.tar.gz.sha256 && rm -rf ckpt && mkdir ckpt && tar xf abst0_ckpt.tar -C ckpt \
      && cd ckpt && sha256sum -c --quiet ../abst0_ckpt.sha256) \
  && echo "== preserved + verified: $OUT ($(wc -l < $OUT/abst0_ckpt.sha256) checkpoints)"
}
cleanup() {
  [ "$DONE" = 1 ] || [ -z "$POD" ] && return
  if [ -z "$S" ]; then echo "== cleanup: removing pod $POD (nothing ran on it)"; runpodctl pod remove $POD >/dev/null; return; fi
  $S "cat /workspace/boot.log /workspace/abst0_p*.log 2>/dev/null" > $OUT/failed_pod.log 2>/dev/null
  # explicit three-way evidence check (Sol): EVIDENCE / NONE only after a successful inspection; anything else
  # (ssh failure, unreadable directory, find error, indeterminate output) -> stop and KEEP the pod
  local ev
  ev=$($S 'if [ ! -e /workspace/abst0 ]; then echo NONE; exit 0; fi
           f=$(find /workspace/abst0 \( -name "*.pt" -o -name "*.json" \) -print -quit) || { echo UNKNOWN; exit 0; }
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
trap cleanup EXIT
pid() { python3 -c 'import json,sys
try: print(json.load(open(sys.argv[1])).get("id",""))
except Exception: pass' "$1"; }
GPUS=("NVIDIA A40" "NVIDIA RTX A6000" "NVIDIA L40S" "NVIDIA RTX 6000 Ada Generation")   # fixed order (Amendment 1)
for round in $(seq ${CB_ROUNDS:-48}); do
  for GPU in "${GPUS[@]}"; do
    if runpodctl pod create --name cb-abst0-p1 --template-id runpod-torch-v280 --gpu-id "$GPU" --cloud-type SECURE \
         --container-disk-in-gb 40 --volume-in-gb 80 --volume-mount-path /workspace --min-cuda-version 13.0 \
         --ssh --wait --wait-timeout 20m > ~/cb_abst0_pod.json 2> ~/cb_abst0_pod.err; then
      POD=$(pid ~/cb_abst0_pod.json); [ -n "$POD" ] && break 2
    fi
    LEFT=$(pid ~/cb_abst0_pod.err); [ -n "$LEFT" ] && { echo "== removing unreachable pod $LEFT"; runpodctl pod remove $LEFT >/dev/null; }
    grep -q 'no longer any instances' ~/cb_abst0_pod.err || { echo "STOP: pod create failed ($GPU):"; cat ~/cb_abst0_pod.err; exit 1; }
  done
  echo "== $(date +%H:%M) round $round: none of the 48 GB cards available (CUDA 13 hosts); retry in 5 min"; sleep 300
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
/workspace/venv/bin/python -m pytest -q -p no:cacheprovider tests/test_abst0.py -m "not slow" > /workspace/pytest.log 2>&1 \
  || { tail -30 /workspace/pytest.log; echo 'STOP: ABST-0 tests fail in this environment'; exit 1; }
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
$S "cat > /workspace/abst0_p1.sh" <<'JOB'
cd /workspace/Continuous_brain_abst0
PY=/workspace/venv/bin/python; M=/workspace/hf/Qwen3-8B; MS=$(cat /workspace/model_manifest.sha256); R=/workspace/abst0
X="$PY scripts/abst0.py"; mkdir -p $R/cal $R/prep
fin() { cd /workspace && tar czf abst0_phase1.tar.gz --exclude='*.pt' abst0 && sha256sum abst0_phase1.tar.gz > abst0_phase1.tar.gz.sha256
  echo "BUNDLE $(cat abst0_phase1.tar.gz.sha256)"; echo "ABST0 P1 DONE exit=$1"; exit $1; }
echo "== calibration (seed 16) $(date -u +%FT%TZ)"
$X prep --seed 16 --model $M --out $R/cal/prep_seed16.json --model-manifest-sha $MS || fin 1
$X evalck --prep $R/cal/prep_seed16.json --model $M --sleep 1 --out $R/cal/eval_N_s1.json --model-manifest-sha $MS || fin 1
for E in 4 8 16; do
  $X train --prep $R/cal/prep_seed16.json --model $M --passes $E --arms P --sleeps 1 --out $R/cal/E$E \
     --calibration --model-manifest-sha $MS || fin 1
  for mode in merged unmerged; do
    $X evalck --prep $R/cal/prep_seed16.json --model $M --sleep 1 --ckpt $R/cal/E$E/cortex_P_s1.pt --mode $mode \
       --out $R/cal/E$E/eval_P_s1_$mode.json --model-manifest-sha $MS || fin 1
  done
done
$X calchoose --dir $R/cal --out $R/cal/calibrate.json --model-manifest-sha $MS; rc=$?
[ $rc = 0 ] || { [ $rc = 4 ] && { echo "STOP: calibration (protocol) - see cal/calibrate.json"; fin 4; }; fin 1; }
for s in 12 13 14 15; do
  echo "== prep seed $s $(date -u +%FT%TZ)"
  $X prep --seed $s --model $M --out $R/prep/prep_seed$s.json --model-manifest-sha $MS || fin 1
done
(cd $R && sha256sum cal/calibrate.json prep/prep_seed*.json prep/audit_sample_seed*.json > PHASE1.sha256)
echo "PIN $(git rev-parse HEAD)" >> $R/PHASE1.sha256; echo "MODEL_MANIFEST $MS" >> $R/PHASE1.sha256
echo "CODE_HASH $($X codehash)" >> $R/PHASE1.sha256
echo "GPU $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)" >> $R/PHASE1.sha256   # phase 2 must match
fin 0
JOB
$S "nohup bash /workspace/abst0_p1.sh > /workspace/abst0_p1.log 2>&1 < /dev/null & sleep 3; echo launched"
t0=$(date +%s)
while :; do
  sleep 120
  if $S "grep -q 'ABST0 P1 DONE' /workspace/abst0_p1.log" 2>/dev/null; then echo "== done"; break; fi
  echo "$(date +%H:%M) $($S "grep -E '^(==|CAL|NEG|PREP|EVAL|TRAIN)' /workspace/abst0_p1.log | tail -1 | cut -c1-150")"
  [ $(( $(date +%s) - t0 )) -gt $((4 * 3600)) ] && { echo "== 4-hour cap reached"; exit 1; }
done
$S "grep -vE 'Loading weights|Writing model' /workspace/abst0_p1.log" | tail -40
$S "cd /workspace && tar cf - abst0_phase1.tar.gz abst0_phase1.tar.gz.sha256 abst0_p1.log boot.log" | tar xf - -C $OUT \
  && (cd $OUT && sha256sum -c abst0_phase1.tar.gz.sha256) && preserve && cp $OUT/abst0_phase1.tar.gz $DL/ \
  && echo "== downloaded + verified: $OUT/abst0_phase1.tar.gz (copy in Downloads); calibration checkpoints in $OUT/ckpt" \
  && runpodctl pod remove $POD >/dev/null && echo "== pod removed" && DONE=1
[ "$DONE" = 1 ] || { echo "== download did not complete"; exit 1; }
EX=$(grep -o 'ABST0 P1 DONE exit=[0-9]*' $OUT/abst0_p1.log | tail -1 | cut -d= -f2); echo "== ABST-0 phase 1 exit: ${EX:-UNKNOWN}"
case "$EX" in 0) exit 0;; 4) exit 4;; *) exit 1;; esac
