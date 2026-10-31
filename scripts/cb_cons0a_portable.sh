#!/usr/bin/env bash
# Run on David's PC (WSL). CONS-0A rerun, VOLUME-FREE (lab/CONS0A_RERUN_LOG.md): for when the volume's data center
# (CA-MTL-1) has no GPU. Same pinned code and the same job as scripts/cb_cons0a_run.sh, with CB_REPLAY=run1 forced
# (Sol's option B: run 1's pinned A40 reconstructions; no reconstruction on this stack). Differences:
#   pod     - any secure-cloud data center; tries the GPUs below, cheapest first, every 5 min (CB_ROUNDS rounds, default
#             96 = 8 h) until one is created; no network volume (80 GB pod volume at /workspace)
#   boot    - fresh venv with PINNED torch==2.14.1 / transformers==5.21.0 (the versions tests/test_cons0a.py passes with;
#             the volume venv's versions could not be read); Qwen/Qwen3-8B at the CLS-0B manifest revision; Qwen3-0.6B
#             tokenizer at the stream-tokenizer path; then tests/test_cons0a.py on CPU in the pinned checkout - any
#             failure STOPs. The unchanged preflight then requires the 12 model files == the CLS-0B manifest, and the
#             job's stream-design / replay checks fail closed if tokenisation differs.
#   results - bundle + log + boot log + ALL 24 cortex checkpoints (sha256-verified on the PC) before the pod is removed
#   identity - every launch is a fresh pod and a fresh campaign (no resume, never mixed environments); platform and
#             pip freeze recorded (GPU.txt, ENV.txt, cb_boot.log); GPU feasibility check (bf16 Qwen3-8B on CUDA, 252 fp32
#             cortex sites, one finite step with nonzero gradients) before calibration
#   cleanup - on any failure the logs (and a bundle, if one exists) are saved to ~/cb_results/cons0a/ and the pod is
#             REMOVED (nothing persistent lives on it); a relaunch starts from scratch.
# Exit: 0 complete; 4 calibration STOP; 1 anything else.        bash cb_cons0a_portable.sh [gpu]
GPUS=("NVIDIA A40" "NVIDIA RTX A6000" "NVIDIA RTX 6000 Ada Generation" "NVIDIA L40S"); [ -n "${1:-}" ] && GPUS=("$1")
K=$HOME/.runpod/ssh/runpodctl-ssh-key
PIN=271b39e653bcfd56f5d9d196914968f6fff1da37                                           # frozen CONS-0A commit
REV=b968826d9c46dd6066d109eabc6255188de91218                                           # Qwen3-8B revision (CLS-0B manifest)
D=/workspace/Continuous_brain_cons0a; OUT=$HOME/cb_results/cons0a; DL=/mnt/c/Users/davei/Downloads
CAP=$((8 * 3600)); POD=""; DONE=0; JOBDONE=0
trap 'if [ "$DONE" != 1 ] && [ -n "$POD" ] && [ "$JOBDONE" = 1 ]; then
  echo "== the job finished but the download was not verified: pod $POD STOPPED, not removed (its /workspace volume keeps"
  echo "   the results and checkpoints; tell Claude). Storage bills about \$0.25/day until it is removed."; runpodctl pod stop $POD >/dev/null
elif [ "$DONE" != 1 ] && [ -n "$POD" ]; then echo "== cleanup: saving logs to $OUT, removing pod $POD"; mkdir -p $OUT
  $S "cat /workspace/cb_boot.log /workspace/cons0a.log 2>/dev/null" > $OUT/failed_pod.log 2>/dev/null
  $S "cat /workspace/results_cons0a.tar.gz" > $OUT/failed_results_cons0a.tar.gz 2>/dev/null || rm -f $OUT/failed_results_cons0a.tar.gz
  runpodctl pod remove $POD >/dev/null; fi' EXIT
pid() { python3 -c 'import json,sys
try: print(json.load(open(sys.argv[1])).get("id",""))
except Exception: pass' "$1"; }
for round in $(seq ${CB_ROUNDS:-96}); do
  for GPU in "${GPUS[@]}"; do
    if runpodctl pod create --name cb-cons0a-portable --template-id runpod-torch-v280 --gpu-id "$GPU" --cloud-type SECURE \
         --container-disk-in-gb 40 --volume-in-gb 80 --volume-mount-path /workspace --min-cuda-version 12.8 \
         --ssh --wait --wait-timeout 20m > ~/cb_cons0a_pod.json 2> ~/cb_cons0a_pod.err; then
      POD=$(pid ~/cb_cons0a_pod.json); [ -n "$POD" ] && break 2
    fi
    LEFT=$(pid ~/cb_cons0a_pod.err)                         # a pod created but not reachable in time: remove it
    [ -n "$LEFT" ] && { echo "== removing unreachable pod $LEFT"; runpodctl pod remove $LEFT >/dev/null; }
    grep -q 'no longer any instances' ~/cb_cons0a_pod.err || { echo "STOP: pod create failed ($GPU):"; cat ~/cb_cons0a_pod.err; exit 1; }
  done
  echo "== $(date +%H:%M) round $round: none of the GPUs available anywhere; retry in 5 min"; sleep 300
done
[ -n "$POD" ] || { echo "STOP: no GPU obtainable after ${CB_ROUNDS:-96} rounds"; exit 1; }
DCN=$(python3 -c 'import json,sys
d=json.load(open(sys.argv[1])); f=lambda o: [v for k,v in o.items() if "datacenter" in k.lower().replace("_","")] + [x for v in o.values() if isinstance(v,dict) for x in f(v)]
print((f(d) or ["unknown"])[0])' ~/cb_cons0a_pod.json 2>/dev/null)
echo "== pod $POD ($GPU, data center ${DCN:-unknown})"
addr() { eval "$(runpodctl ssh info $POD 2>/dev/null | python3 -c 'import sys,json
try: d=json.load(sys.stdin); print("IP=%s PORT=%s" % (d["ip"], d["port"]))
except Exception: pass')"; }
addr; S="ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 -i $K -p $PORT root@$IP"
echo "== boot: venv (pinned), model + tokenizer download, test suite (10-15 min)"
$S "D=$D PIN=$PIN REV=$REV bash -s" <<'BOOT' 2>&1 | tee ~/cb_cons0a_boot.log | grep -E '^(ENV|BOOT|STOP|[0-9]+ passed|FAILED)'
set -Eeuo pipefail
exec > >(tee /workspace/cb_boot.log) 2>&1
rm -f /workspace/cb_portable_boot.ok
mkdir -p /workspace/Continuous_brain/models /workspace/hf
cd /workspace/Continuous_brain
[ -x .venv/bin/python ] || python3 -m venv .venv || { python3 -m pip install -q virtualenv && python3 -m virtualenv -q .venv; }
.venv/bin/python -m pip install -q --upgrade pip
DRV=$(nvidia-smi | grep -o 'CUDA Version: [0-9.]*' | awk '{print $3}'); echo "ENV host driver supports CUDA $DRV"
if python3 -c "import sys; sys.exit(0 if tuple(map(int, '${DRV:-0}'.split('.'))) >= (13, 0) else 1)"; then
  .venv/bin/python -m pip install -q torch==2.14.1                       # PyPI build = CUDA 13.0
else
  .venv/bin/python -m pip install -q torch==2.14.1 --index-url https://download.pytorch.org/whl/cu128 \
    || .venv/bin/python -m pip install -q torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
fi
.venv/bin/python -m pip install -q transformers==5.21.0 pytest
.venv/bin/python -c "import torch, transformers, sys; assert torch.__version__.startswith('2.14.1') and transformers.__version__ == '5.21.0'; assert torch.cuda.is_available(), 'CUDA not available'; print('ENV python', sys.version.split()[0], 'torch', torch.__version__, 'cuda', torch.version.cuda, 'transformers', transformers.__version__, 'gpu', torch.cuda.get_device_name(0))" \
  || { echo 'STOP: pinned environment unusable'; exit 1; }
.venv/bin/python - <<PY || { echo 'STOP: model/tokenizer download failed'; exit 1; }
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3-8B", revision="$REV", local_dir="/workspace/hf/Qwen3-8B",
                  allow_patterns=["*.safetensors", "*.json", "*.txt"])
snapshot_download("Qwen/Qwen3-0.6B", local_dir="/workspace/Continuous_brain/models/qwen3-0.6b",
                  allow_patterns=["tokenizer*", "vocab.json", "merges.txt", "config.json"])
print("BOOT downloads done")
PY
[ -d $D ] || git clone -q --depth 200 -b being/v3-bind https://github.com/razerblade72611/Continuous_brain $D
cd $D && git checkout -q --detach $PIN && [ "$(git rev-parse HEAD)" = "$PIN" ]
CUDA_VISIBLE_DEVICES="" /workspace/Continuous_brain/.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_cons0a.py \
  > /workspace/cb_pytest.log 2>&1 || { tail -40 /workspace/cb_pytest.log; echo 'STOP: CONS-0A tests fail in this environment'; exit 1; }
tail -1 /workspace/cb_pytest.log
grep -qE '^12 passed' <(tail -1 /workspace/cb_pytest.log) || { echo 'STOP: CONS-0A tests did not all run (expected 12 passed)'; exit 1; }
cd $D && /workspace/Continuous_brain/.venv/bin/python - <<'PY' || { echo 'STOP: GPU feasibility check failed'; exit 1; }
import importlib.util, math, torch                          # Sol: real-model GPU feasibility before calibration
spec = importlib.util.spec_from_file_location("c", "scripts/cons0a.py"); C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)
assert torch.cuda.is_available() and C._dev() == "cuda", "CUDA not selected"
model, tok = C.load_model("/workspace/hf/Qwen3-8B", "cuda")
assert next(model.parameters()).dtype == torch.bfloat16, "base model not bf16"
cx = C.Cortex(model, seed=0)
n = len(cx.sites); assert n == model.config.num_hidden_layers * len(C.TARGETS), f"{n} cortex sites"
assert all(A.dtype == torch.float32 and B.dtype == torch.float32 for _, _, A, B in cx.sites), "cortex not fp32"
ids, k = C.qa_ids(tok, "Where does Ann Pell live?", "Leston")
x = torch.tensor([ids], device="cuda"); lab = x.clone(); lab[0, :-k] = -100
C.cx_model_train(model); loss = model(input_ids=x, labels=lab).loss; loss.backward()
g = sum(float(B.grad.abs().sum()) for _, _, _, B in cx.sites if B.grad is not None)
assert math.isfinite(float(loss)) and g > 0, (float(loss), g)
print(f"BOOT GPU feasibility OK: {torch.cuda.get_device_name(0)}, {n} cortex sites, bf16 base / fp32 cortex, "
      f"loss {float(loss):.3f}, |grad B| {g:.3g}")
PY
find $D -name __pycache__ -prune -exec rm -rf {} +
touch /workspace/cb_portable_boot.ok; echo "BOOT OK"
BOOT
$S "[ -f /workspace/cb_portable_boot.ok ]" || { echo "STOP: boot failed - see $OUT/failed_pod.log"; exit 1; }
$S "VOL=none D=$D PIN=$PIN CB_UNLOCK=${CB_UNLOCK:-0} CB_REPLAY=run1 bash -s" <<'REMOTE' || { echo "STOP: preflight/launch failed"; exit 1; }
set -Eeuo pipefail
[ -f /workspace/cb_portable_boot.ok ] || { echo 'STOP: portable boot not completed'; exit 1; }
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
rm -f /workspace/results_cons0a.tar.gz /workspace/results_cons0a.tar.gz.sha256 /workspace/cons0a_ckpt.tar /workspace/cons0a_ckpt.sha256
R=results/cons0a; M=/workspace/hf/Qwen3-8B; PY=.venv/bin/python
case "${CB_REPLAY:-prep}" in prep|run1) ;; *) echo "STOP: CB_REPLAY must be prep or run1"; echo 'CONS-0A JOB DONE exit=1'; exit 1;; esac
HEADC="$(git rev-parse HEAD) replay=${CB_REPLAY:-prep}"      # resume only results made by THIS commit and replay mode
if [ -d $R ] && [ "$(cat $R/PIN 2>/dev/null)" != "$HEADC" ]; then
  OLD=results/cons0a_stale_$(cat $R/PIN 2>/dev/null | cut -c1-7)_$(date -u +%Y%m%dT%H%M%S); mv $R $OLD
  find $OLD -name '*.pt' -delete; echo "== results from another commit moved to $OLD (checkpoints removed); not resumed"
fi
mkdir -p $R; echo "$HEADC" > $R/PIN
nvidia-smi --query-gpu=name --format=csv,noheader > $R/GPU.txt 2>/dev/null || echo unknown > $R/GPU.txt
$PY -m pip freeze --all > $R/ENV.txt 2>/dev/null; $PY --version >> $R/ENV.txt 2>&1
echo "== GPU: $(cat $R/GPU.txt); replay mode: ${CB_REPLAY:-prep}"
finish() { echo "== end $(date -u +%FT%TZ)"
  tar czf /workspace/results_cons0a.tar.gz --exclude='*.pt' $R
  find $R -name '*.pt' | sort > /workspace/ckpt.list; (xargs -r sha256sum < /workspace/ckpt.list) > /workspace/cons0a_ckpt.sha256
  tar cf /workspace/cons0a_ckpt.tar -T /workspace/ckpt.list && echo "CHECKPOINTS $(wc -l < /workspace/ckpt.list) files"
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
    $PY - $O/replay.json results/cls0b_final/seed${s}_bank.pt <<'PV' || { echo "STOP: run-1 replay for seed $s: bank or model manifest differs from the verified originals"; finish 1; }
import hashlib, json, sys                                     # provenance (Sol): replay bank + model == preflight-verified ones
r = json.load(open(sys.argv[1]))
ok = r["bank_sha256"] == hashlib.sha256(open(sys.argv[2], "rb").read()).hexdigest() and \
    r["model_manifest"] == json.load(open("results/cls0b_final/seed2.json"))["model_manifest"]["files"]
sys.exit(0 if ok else 1)
PV
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
  if $S "grep -q 'CONS-0A JOB DONE' /workspace/cons0a.log" 2>/dev/null; then echo "== done"; JOBDONE=1; break; fi
  echo "$(date +%H:%M) running: $($S "tail -n 1 /workspace/cons0a.log | cut -c1-110" 2>/dev/null)"
  [ $(( $(date +%s) - t0 )) -gt $CAP ] && { echo "== 8-hour cap reached"; exit 1; }
done
$S "grep -v 'Loading weights\|generation flags\|attention mask' /workspace/cons0a.log"
mkdir -p $OUT
$S "cd /workspace && tar cf - results_cons0a.tar.gz results_cons0a.tar.gz.sha256 cons0a.log cb_boot.log cons0a_ckpt.tar cons0a_ckpt.sha256" | tar xf - -C $OUT \
  && (cd $OUT && sha256sum -c results_cons0a.tar.gz.sha256) \
  && (cd $OUT && rm -rf ckpt && mkdir ckpt && tar xf cons0a_ckpt.tar -C ckpt && cd ckpt && sha256sum -c --quiet ../cons0a_ckpt.sha256 \
      && [ $(wc -l < ../cons0a_ckpt.sha256) = 24 ] && echo "== 24 cortex checkpoints verified: $OUT/ckpt") \
  && cp $OUT/results_cons0a.tar.gz $DL/ \
  && echo "== downloaded + verified: $OUT (copy in Downloads)" \
  && runpodctl pod remove $POD >/dev/null && echo "== pod removed" && DONE=1
[ "$DONE" = 1 ] || { echo "== download/verification did not complete"; exit 1; }
E=$(grep -o 'CONS-0A JOB DONE exit=[0-9]*' $OUT/cons0a.log | tail -n 1 | cut -d= -f2)
echo "== CONS-0A exit: ${E:-UNKNOWN}"
case "$E" in 0) exit 0;; 4) exit 4;; *) exit 1;; esac
