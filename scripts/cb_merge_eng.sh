#!/usr/bin/env bash
# Run on David's PC (WSL). MERGE-ENG (Sol GO 2026-10-31; lab/CONS0A_RUN2_REPORT.md): read-only engineering check of
# merged-vs-unmerged equivalence on the 24 CONS-0A run-2 cortex checkpoints in ~/cb_results/cons0a/ckpt. No training.
#   1. verifies the local checkpoints against ~/cb_results/cons0a/cons0a_ckpt.sha256 (run 2's own list)
#   2. A40 pod (run 2's GPU), any data center, retried every 5 min (CB_ROUNDS, default 48 = 4 h); CUDA >= 13.0 host
#   3. boot: venv = run 2's exact pip freeze (results/cons0a_run2/ENV.txt); Qwen3-8B at the CLS-0B revision and the
#      stream tokenizer, both required to equal the CLS-0B manifest
#   4. uploads the checkpoints (~4 GB; time depends on your upload speed) and re-verifies their sha256 on the pod
#   5. scripts/merge_eng.py (7 precision configurations x 24 checkpoints; ~2.5 h), then downloads
#      results/merge_eng/merge_eng.json + log to ~/cb_results/merge_eng and removes the pod (also on failure, logs saved)
#   bash cb_merge_eng.sh
PIN=48764b253371348353daf215b48947deb45045c7                                                                         # commit with merge_eng.py
REV=b968826d9c46dd6066d109eabc6255188de91218
K=$HOME/.runpod/ssh/runpodctl-ssh-key; CK=$HOME/cb_results/cons0a; OUT=$HOME/cb_results/merge_eng
D=/workspace/Continuous_brain_me; POD=""; DONE=0; mkdir -p $OUT
[ -d $CK/ckpt/results/cons0a ] && [ -f $CK/cons0a_ckpt.sha256 ] || { echo "STOP: run-2 checkpoints not found in $CK/ckpt"; exit 1; }
echo "== verifying the 24 local checkpoints"
(cd $CK/ckpt && sha256sum -c --quiet ../cons0a_ckpt.sha256) && [ $(wc -l < $CK/cons0a_ckpt.sha256) = 24 ] \
  || { echo "STOP: local checkpoints do not match run 2's sha256 list"; exit 1; }
trap 'if [ "$DONE" != 1 ] && [ -n "$POD" ]; then echo "== cleanup: saving log to $OUT, removing pod $POD"
  $S "cat /workspace/me_boot.log /workspace/merge_eng.log 2>/dev/null" > $OUT/failed_pod.log 2>/dev/null
  $S "cat $D/results/merge_eng/merge_eng.json" > $OUT/partial_merge_eng.json 2>/dev/null || rm -f $OUT/partial_merge_eng.json
  runpodctl pod remove $POD >/dev/null; fi' EXIT
pid() { python3 -c 'import json,sys
try: print(json.load(open(sys.argv[1])).get("id",""))
except Exception: pass' "$1"; }
for round in $(seq ${CB_ROUNDS:-48}); do
  if runpodctl pod create --name cb-merge-eng --template-id runpod-torch-v280 --gpu-id "NVIDIA A40" --cloud-type SECURE \
       --container-disk-in-gb 40 --volume-in-gb 80 --volume-mount-path /workspace --min-cuda-version 13.0 \
       --ssh --wait --wait-timeout 20m > ~/cb_me_pod.json 2> ~/cb_me_pod.err; then
    POD=$(pid ~/cb_me_pod.json); [ -n "$POD" ] && break
  fi
  LEFT=$(pid ~/cb_me_pod.err); [ -n "$LEFT" ] && { echo "== removing unreachable pod $LEFT"; runpodctl pod remove $LEFT >/dev/null; }
  grep -q 'no longer any instances' ~/cb_me_pod.err || { echo "STOP: pod create failed:"; cat ~/cb_me_pod.err; exit 1; }
  echo "== $(date +%H:%M) round $round: no A40 available; retry in 5 min"; sleep 300
done
[ -n "$POD" ] || { echo "STOP: no A40 after ${CB_ROUNDS:-48} rounds"; exit 1; }
echo "== pod $POD (NVIDIA A40)"
eval "$(runpodctl ssh info $POD | python3 -c 'import sys,json;d=json.load(sys.stdin);print("IP=%s PORT=%s" % (d["ip"], d["port"]))')"
S="ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -i $K -p $PORT root@$IP"
echo "== boot: run 2's exact environment, model + tokenizer (10-15 min)"
$S "D=$D PIN=$PIN REV=$REV bash -s" <<'BOOT' 2>&1 | grep -E '^(ENV|BOOT|STOP)'
set -Eeuo pipefail
exec > >(tee /workspace/me_boot.log) 2>&1
git clone -q --depth 200 -b being/v3-bind https://github.com/razerblade72611/Continuous_brain $D
cd $D && git checkout -q --detach $PIN && [ "$(git rev-parse HEAD)" = "$PIN" ]
python3 -m venv /workspace/venv || { python3 -m pip install -q virtualenv && python3 -m virtualenv -q /workspace/venv; }
grep -v '^Python ' results/cons0a_run2/ENV.txt | grep -v '^-e ' > /workspace/req.txt
/workspace/venv/bin/python -m pip install -q -r /workspace/req.txt || { echo 'STOP: could not install run 2 environment'; exit 1; }
/workspace/venv/bin/python -c "import torch, transformers, sys; assert torch.cuda.is_available(); print('ENV python', sys.version.split()[0], 'torch', torch.__version__, 'transformers', transformers.__version__, 'gpu', torch.cuda.get_device_name(0))"
/workspace/venv/bin/python - <<PY || { echo 'STOP: download or identity check failed'; exit 1; }
import hashlib, importlib.util, json
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3-8B", revision="$REV", local_dir="/workspace/hf/Qwen3-8B", allow_patterns=["*.safetensors", "*.json", "*.txt"])
snapshot_download("Qwen/Qwen3-0.6B", local_dir="/workspace/tok", allow_patterns=["tokenizer*", "vocab.json", "merges.txt", "config.json"])
spec = importlib.util.spec_from_file_location("b", "scripts/cls0c_b.py"); b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
m = json.load(open("results/cls0b_final/seed2.json"))["model_manifest"]["files"]
assert b.manifest("/workspace/hf/Qwen3-8B") == m, "model files differ from the CLS-0B manifest"
for f in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
    assert hashlib.sha256(open(f"/workspace/tok/{f}", "rb").read()).hexdigest() == m[f], f
print("BOOT model + tokenizer == CLS-0B manifest")
PY
echo "BOOT OK"
BOOT
$S "grep -q 'BOOT OK' /workspace/me_boot.log" || { echo "STOP: boot failed - see $OUT/failed_pod.log"; exit 1; }
echo "== uploading the checkpoints (~4 GB)"
tar cf - -C $CK/ckpt results | $S "mkdir -p /workspace/ck && tar xf - -C /workspace/ck" \
  && cat $CK/cons0a_ckpt.sha256 | $S "cd /workspace/ck && sha256sum -c --quiet -" \
  && echo "== checkpoints uploaded and verified on the pod" || { echo "STOP: checkpoint upload/verification failed"; exit 1; }
$S "cd $D && nohup /workspace/venv/bin/python scripts/merge_eng.py --run2 results/cons0a_run2 --ckpts /workspace/ck/results/cons0a \
  --model /workspace/hf/Qwen3-8B --stream-tokenizer /workspace/tok --out results/merge_eng/merge_eng.json \
  > /workspace/merge_eng.log 2>&1 < /dev/null & sleep 3; echo launched"
t0=$(date +%s)
while :; do
  sleep 120
  if $S "grep -q 'MERGE-ENG DONE' /workspace/merge_eng.log" 2>/dev/null; then echo "== done"; break; fi
  if ! $S "pgrep -f '[m]erge_eng.py' >/dev/null" 2>/dev/null; then echo "STOP: merge_eng.py exited early:"; $S "tail -30 /workspace/merge_eng.log"; exit 1; fi
  echo "$(date +%H:%M) $($S "grep -E '^[0-9]{2}:[0-9]{2}:[0-9]{2} [AB] ' /workspace/merge_eng.log | wc -l")/48 checkpoint-phases done: $($S "grep -E '^[0-9]{2}:[0-9]{2}:[0-9]{2} [AB] ' /workspace/merge_eng.log | tail -1 | cut -c1-150")"
  [ $(( $(date +%s) - t0 )) -gt $((5 * 3600)) ] && { echo "== 5-hour cap reached"; exit 1; }
done
$S "cat $D/results/merge_eng/merge_eng.json" > $OUT/merge_eng.json && $S "cat /workspace/merge_eng.log" > $OUT/merge_eng.log \
  && python3 -c "import json;json.load(open('$OUT/merge_eng.json'))" && grep -q 'MERGE-ENG DONE' $OUT/merge_eng.log \
  && cp $OUT/merge_eng.json /mnt/c/Users/davei/Downloads/ && echo "== downloaded: $OUT/merge_eng.json (copy in Downloads)" \
  && runpodctl pod remove $POD >/dev/null && echo "== pod removed" && DONE=1
[ "$DONE" = 1 ] || { echo "== download did not complete"; exit 1; }
