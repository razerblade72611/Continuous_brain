#!/usr/bin/env bash
# Run on David's PC (WSL). ABST-0 PHASE 2 (lab/ABST0_PROTOCOL.md): development training + fp32 evaluation + analysis.
# Requires the committed phase-1 artifacts (results/abst0/phase1/: calibrate.json, prep_seed12-15.json, audit samples,
# PHASE1.sha256) AND Claude's completed blind-audit labels (audit_labels_seed12-15.json) at the pinned commit.
#   job : verify PHASE1.sha256 + the model manifest it recorded; chosen E from the successful calibration; per seed:
#         train P/U/G (audit + calibration bound) -> N x5 + (P,U,G) x 5 sleeps x {merged, unmerged} fp32 evaluations ->
#         audit statistics; then the validated analysis
#   out : ~/cb_results/abst0/phase2/: results bundle (no .pt) + ALL 60 cortex checkpoints, sha256-verified, before the
#         pod is removed (if the job finished but the download did not verify, the pod is STOPPED, not removed)
#   bash cb_abst0_p2.sh
PIN=__PIN__
GPU=__GPU__                                              # the card type phase 1 recorded (Amendment 1); set when pinned
REV=b968826d9c46dd6066d109eabc6255188de91218
K=$HOME/.runpod/ssh/runpodctl-ssh-key; OUT=$HOME/cb_results/abst0/phase2; DL=/mnt/c/Users/davei/Downloads
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
[[ $GPU == NVIDIA* ]] || { echo "STOP: launcher GPU is not set to phase 1's recorded card type"; exit 1; }
trap cleanup EXIT
pid() { python3 -c 'import json,sys
try: print(json.load(open(sys.argv[1])).get("id",""))
except Exception: pass' "$1"; }
for round in $(seq ${CB_ROUNDS:-48}); do
  if runpodctl pod create --name cb-abst0-p2 --template-id runpod-torch-v280 --gpu-id "$GPU" --cloud-type SECURE \
       --container-disk-in-gb 40 --volume-in-gb 80 --volume-mount-path /workspace --min-cuda-version 13.0 \
       --ssh --wait --wait-timeout 20m > ~/cb_abst0_pod.json 2> ~/cb_abst0_pod.err; then
    POD=$(pid ~/cb_abst0_pod.json); [ -n "$POD" ] && break
  fi
  LEFT=$(pid ~/cb_abst0_pod.err); [ -n "$LEFT" ] && { echo "== removing unreachable pod $LEFT"; runpodctl pod remove $LEFT >/dev/null; }
  grep -q 'no longer any instances' ~/cb_abst0_pod.err || { echo "STOP: pod create failed:"; cat ~/cb_abst0_pod.err; exit 1; }
  echo "== $(date +%H:%M) round $round: no $GPU available (CUDA 13 hosts); retry in 5 min"; sleep 300
done
[ -n "$POD" ] || { echo "STOP: no $GPU after ${CB_ROUNDS:-48} rounds"; exit 1; }
echo "== pod $POD ($GPU)"
eval "$(runpodctl ssh info $POD | python3 -c 'import sys,json;d=json.load(sys.stdin);print("IP=%s PORT=%s" % (d["ip"], d["port"]))')"
S="ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -i $K -p $PORT root@$IP"   # a dead connection fails in ~3 min
echo "== boot: run 2's exact environment, model + tokenizer, pure tests (10-15 min)"
$S "cat > /workspace/boot.sh" <<'BOOT'
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
# boot runs detached on the pod (a dropped connection cannot stall it); poll its log for BOOT OK / STOP, 40 min max
$S "D=$D PIN=$PIN REV=$REV nohup bash /workspace/boot.sh > /dev/null 2>&1 < /dev/null & echo boot started" || { echo "STOP: could not start boot"; exit 1; }
shown=0; t0=$(date +%s)
while :; do
  sleep 30
  L=$($S "grep -E '^(ENV|BOOT|STOP|[0-9]+ passed|FAILED)' /workspace/boot.log 2>/dev/null" 2>/dev/null) || { echo "$(date +%H:%M) (pod not reachable this time; retrying)"; continue; }
  n=$(printf '%s\n' "$L" | grep -c . ); [ "$n" -gt "$shown" ] && { printf '%s\n' "$L" | tail -n $((n - shown)); shown=$n; }
  printf '%s\n' "$L" | grep -q '^BOOT OK' && break
  printf '%s\n' "$L" | grep -q '^STOP' && { echo "STOP: boot failed - see $OUT/failed_pod.log"; exit 1; }
  [ $(( $(date +%s) - t0 )) -gt 2400 ] && { echo "STOP: boot did not finish in 40 min - see $OUT/failed_pod.log"; exit 1; }
done
$S "cat > /workspace/abst0_p2.sh" <<'JOB'
cd /workspace/Continuous_brain_abst0
PY=/workspace/venv/bin/python; M=/workspace/hf/Qwen3-8B; MS=$(cat /workspace/model_manifest.sha256); R=/workspace/abst0
P1=results/abst0/phase1; X="$PY scripts/abst0.py"; mkdir -p $R
fin() { cd /workspace && tar czf abst0_phase2.tar.gz --exclude='*.pt' abst0 && sha256sum abst0_phase2.tar.gz > abst0_phase2.tar.gz.sha256
  (cd /workspace && find abst0 -name '*.pt' | sort > ckpt.list && xargs -r sha256sum < ckpt.list > abst0_ckpt.sha256 \
     && tar cf abst0_ckpt.tar -T ckpt.list && echo "CHECKPOINTS $(wc -l < ckpt.list) files")
  echo "BUNDLE $(cat abst0_phase2.tar.gz.sha256)"; echo "ABST0 P2 DONE exit=$1"; exit $1; }
echo "== verifying phase-1 artifacts $(date -u +%FT%TZ)"
(cd $P1 && grep -v '^PIN \|^MODEL_MANIFEST \|^CODE_HASH \|^GPU ' PHASE1.sha256 | sha256sum -c --quiet -) || { echo "STOP: phase-1 artifacts differ from PHASE1.sha256"; fin 1; }
G1=$(grep '^GPU ' $P1/PHASE1.sha256 | cut -d' ' -f2-); G2=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
[ -n "$G1" ] && [ -n "$G2" ] && [ "$G1" = "$G2" ] \
  || { echo "STOP: GPU names missing or different (phase 1: '$G1', this pod: '$G2')"; fin 1; }   # two empty names never match
[ "$(grep '^CODE_HASH ' $P1/PHASE1.sha256 | cut -d' ' -f2)" = "$($X codehash)" ] \
  || { echo "STOP: phase 2's executable/data hash differs from the frozen phase-1 hash"; fin 1; }
echo "{\"phase1_pin\": \"$(grep '^PIN ' $P1/PHASE1.sha256 | cut -d' ' -f2)\", \"phase2_pin\": \"$(git rev-parse HEAD)\", \"code_hash\": \"$($X codehash)\", \"note\": \"phase 2 runs at a later commit that adds the audited phase-1 artifacts; the frozen executable/data hash is identical\"}" > $R/PINS.json
[ "$(grep '^MODEL_MANIFEST ' $P1/PHASE1.sha256 | cut -d' ' -f2)" = "$MS" ] || { echo "STOP: phase 1 used a different model"; fin 1; }
E=$($PY -c "import json,sys; c=json.load(open('$P1/cal/calibrate.json')); sys.exit(1) if c['stop'] or c['errors'] else print(c['chosen_E'])") \
  || { echo "STOP: calibration was not successful"; fin 1; }
echo "calibration: E=$E"
for s in 12 13 14 15; do
  O=$R/seed$s; mkdir -p $O; cp $P1/prep/prep_seed$s.json $O/
  for f in audit_sample audit_labels; do [ -f $P1/prep/${f}_seed$s.json ] || { echo "STOP: $f for seed $s missing"; fin 1; }; done
  echo "== seed $s train $(date -u +%FT%TZ)"
  $X train --prep $O/prep_seed$s.json --model $M --passes $E --out $O --model-manifest-sha $MS \
     --calibration-result $P1/cal/calibrate.json --audit-sample $P1/prep/audit_sample_seed$s.json \
     --audit-labels $P1/prep/audit_labels_seed$s.json || fin 1
  for L in 1 2 3 4 5; do
    $X evalck --prep $O/prep_seed$s.json --model $M --sleep $L --out $O/eval_N_s$L.json --model-manifest-sha $MS || fin 1
    for arm in P U G; do for mode in merged unmerged; do
      $X evalck --prep $O/prep_seed$s.json --model $M --sleep $L --ckpt $O/cortex_${arm}_s$L.pt --mode $mode \
         --out $O/eval_${arm}_s${L}_$mode.json --model-manifest-sha $MS || fin 1
    done; done
  done
  $X auditstats --prep $O/prep_seed$s.json --audit-sample $P1/prep/audit_sample_seed$s.json \
     --audit-labels $P1/prep/audit_labels_seed$s.json --train $O/train_seed$s.json --out $O/audit_stats_seed$s.json || fin 1
done
$X analyze --dir $R --model-manifest-sha $MS --out $R/analysis.json || fin 1
fin 0
JOB
$S "nohup bash /workspace/abst0_p2.sh > /workspace/abst0_p2.log 2>&1 < /dev/null & sleep 3; echo launched"
t0=$(date +%s)
while :; do
  sleep 120
  if $S "grep -q 'ABST0 P2 DONE' /workspace/abst0_p2.log" 2>/dev/null; then echo "== done"; break; fi
  echo "$(date +%H:%M) $($S "grep -E '^(==|EVAL|TRAIN|calibration)' /workspace/abst0_p2.log | tail -1 | cut -c1-150")"
  [ $(( $(date +%s) - t0 )) -gt $((14 * 3600)) ] && { echo "== 14-hour cap reached"; exit 1; }
done
$S "grep -vE 'Loading weights|Writing model' /workspace/abst0_p2.log" | grep -E 'verdict|STOP|calibration|CHECKPOINTS|BUNDLE|DONE|== seed' | tail -30
$S "cd /workspace && tar cf - abst0_phase2.tar.gz abst0_phase2.tar.gz.sha256 abst0_p2.log boot.log abst0_ckpt.tar abst0_ckpt.sha256" | tar xf - -C $OUT \
  && (cd $OUT && sha256sum -c abst0_phase2.tar.gz.sha256) \
  && (cd $OUT && rm -rf ckpt && mkdir ckpt && tar xf abst0_ckpt.tar -C ckpt && cd ckpt && sha256sum -c --quiet ../abst0_ckpt.sha256 \
      && [ $(wc -l < ../abst0_ckpt.sha256) = 60 ] && echo "== 60 cortex checkpoints verified: $OUT/ckpt") \
  && cp $OUT/abst0_phase2.tar.gz $DL/ && echo "== downloaded + verified: $OUT/abst0_phase2.tar.gz (copy in Downloads)" \
  && runpodctl pod remove $POD >/dev/null && echo "== pod removed" && DONE=1
[ "$DONE" = 1 ] || { echo "== download did not complete"; exit 1; }
EX=$(grep -o 'ABST0 P2 DONE exit=[0-9]*' $OUT/abst0_p2.log | tail -1 | cut -d= -f2); echo "== ABST-0 phase 2 exit: ${EX:-UNKNOWN}"
[ "$EX" = 0 ] && exit 0 || exit 1
