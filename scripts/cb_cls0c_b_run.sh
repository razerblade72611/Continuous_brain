#!/usr/bin/env bash
# Run on David's PC (WSL). CLS-0C Phase B (read-only development comparison, lab/CLS0C_B_PROTOCOL.md): create an A40 pod
# on volume 29qhoqghqf (CA-MTL-1), use a SEPARATE checkout pinned to the Phase B commit (the CLS-0B checkout and its results
# on the volume are not touched), verify the committed CLS-0B inputs, run Phase B (real-latency preflight inside: exit 4 if
# the projection exceeds 2.5 h; launcher cap 4 h), download + verify the bundle to ~/cb_results/cls0c_b, remove the pod.
# Exit: 0 complete, 4 preflight stop, 1 operational failure.
#   bash cb_cls0c_b_run.sh [gpu="NVIDIA A40"]
GPU=${1:-NVIDIA A40}; VOL=29qhoqghqf; DC=CA-MTL-1; K=$HOME/.runpod/ssh/runpodctl-ssh-key
PIN=428d02a22ad659cd7f6edb779bc488ddb5255881                   # Phase B commit (scripts/cls0c_b.py)
D=/workspace/Continuous_brain_c0cb; CAP=$((4 * 3600)); t0=$(date +%s)
runpodctl network-volume get $VOL >/dev/null || { echo "STOP: volume $VOL not found"; exit 1; }
runpodctl pod create --name cb-cls0c-b --template-id runpod-torch-v280 --gpu-id "$GPU" --data-center-ids $DC \
  --network-volume-id $VOL --volume-mount-path /workspace --ssh --wait --wait-timeout 20m > ~/cb_c0cb_pod.json \
  || { echo "STOP: pod create failed"; cat ~/cb_c0cb_pod.json; exit 1; }
POD=$(python3 -c 'import json;print(json.load(open("'"$HOME"'/cb_c0cb_pod.json"))["id"])') || exit 1
DONE=0                                                       # any unsuccessful exit after this point stops the pod
trap 'if [ "$DONE" != 1 ] && [ -n "$POD" ]; then echo "== cleanup: stopping pod $POD (files stay on the volume)"; runpodctl pod stop $POD; fi' EXIT
echo "== pod $POD ($GPU, $DC)"
addr() { eval "$(runpodctl ssh info $POD 2>/dev/null | python3 -c 'import sys,json
try: d=json.load(sys.stdin); print("IP=%s PORT=%s" % (d["ip"], d["port"]))
except Exception: pass')"; }
addr; S="ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 -i $K -p $PORT root@$IP"
$S "VOL=$VOL D=$D PIN=$PIN bash -s" <<'REMOTE' || { echo "STOP: setup/launch failed - removing pod"; runpodctl pod remove $POD && POD=""; exit 1; }
set -Eeuo pipefail
findmnt -no SOURCE /workspace | grep -q networkvolumes/$VOL
if pgrep -f '[s]cripts/cls0' >/dev/null; then echo 'STOP: a CLS process is running in this pod'; exit 1; fi
[ -d $D ] || git clone -q --depth 50 -b being/v3-bind https://github.com/razerblade72611/Continuous_brain $D
cd $D
if ! git diff --quiet || ! git diff --cached --quiet; then echo 'STOP: tracked files modified in the Phase B checkout'; git status --short; exit 1; fi
if [ -e results/cls0c_b ]; then mv results/cls0c_b results/cls0c_b.$(date -u +%Y%m%dT%H%M%SZ); echo 'earlier Phase B output moved aside'; fi
git fetch -q --depth 50 origin being/v3-bind
git cat-file -e $PIN^{commit}
git checkout -q --detach $PIN
[ "$(git rev-parse HEAD)" = "$PIN" ] || { echo 'STOP: HEAD is not the pinned Phase B commit'; exit 1; }
echo "Phase B code at: $(git log --oneline -1)"
(cd results/cls0b_final && sha256sum -c --quiet FILES.sha256) || { echo 'STOP: CLS-0B inputs do not match FILES.sha256'; exit 1; }
ln -sfn /workspace/Continuous_brain/.venv .venv
for m in Qwen3-8B Qwen3-Embedding-0.6B; do [ -f /workspace/hf/$m/config.json ] || { echo "STOP: model $m missing"; exit 1; }; done
rm -f /workspace/cls0c_b.log /workspace/results_cls0c_b.tar.gz /workspace/results_cls0c_b.tar.gz.sha256
cat > /workspace/cls0c_b_job.sh <<'JOB'
cd /workspace/Continuous_brain_c0cb
timeout 13500 .venv/bin/python scripts/cls0c_b.py --results results/cls0b_final --model /workspace/hf/Qwen3-8B \
    --external /workspace/hf/Qwen3-Embedding-0.6B --out results/cls0c_b; e=$?
echo "Phase B exit $e"
tar czf /workspace/results_cls0c_b.tar.gz results/cls0c_b
sha256sum /workspace/results_cls0c_b.tar.gz | sed 's|/workspace/||' > /workspace/results_cls0c_b.tar.gz.sha256 && sync
echo "BUNDLE $(cat /workspace/results_cls0c_b.tar.gz.sha256)"
echo "CLS-0C-B JOB DONE exit=$e"
JOB
nohup bash /workspace/cls0c_b_job.sh > /workspace/cls0c_b.log 2>&1 < /dev/null &
sleep 5; echo "PHASE B LAUNCHED"
REMOTE
echo "== waiting for Phase B (checks every 2 min, cap 4 h)"
while :; do
  sleep 120; addr; S="ssh -o ConnectTimeout=10 -i $K -p $PORT root@$IP"
  if $S "grep -q 'CLS-0C-B JOB DONE' /workspace/cls0c_b.log" 2>/dev/null; then echo "== done"; break; fi
  echo "$(date +%H:%M) running: $($S "tail -n 1 /workspace/cls0c_b.log | cut -c1-100" 2>/dev/null)"
  [ $(( $(date +%s) - t0 )) -gt $CAP ] && { echo "== 4-hour cap reached - stopping pod (files stay on the volume)"; runpodctl pod stop $POD; POD=""; exit 1; }
done
$S "grep -v 'Loading weights\|generation flags\|attention mask' /workspace/cls0c_b.log"
mkdir -p ~/cb_results/cls0c_b
$S "cd /workspace && tar cf - results_cls0c_b.tar.gz results_cls0c_b.tar.gz.sha256 cls0c_b.log" | tar xf - -C ~/cb_results/cls0c_b \
  && (cd ~/cb_results/cls0c_b && sha256sum -c results_cls0c_b.tar.gz.sha256) \
  && cp ~/cb_results/cls0c_b/results_cls0c_b.tar.gz /mnt/c/Users/davei/Downloads/ \
  && echo "== downloaded + verified: ~/cb_results/cls0c_b (copy in Downloads)" \
  && runpodctl pod remove $POD >/dev/null && echo "== pod removed" && DONE=1
[ "$DONE" = 1 ] || { echo "== download/verification did not complete - pod stopped by cleanup, files stay on the volume"; exit 1; }
E=$(grep -o 'CLS-0C-B JOB DONE exit=[0-9]*' ~/cb_results/cls0c_b/cls0c_b.log | tail -n 1 | cut -d= -f2)
echo "== CLS-0C-B exit: ${E:-UNKNOWN}"
case "$E" in 0) exit 0;; 4) exit 4;; *) exit 1;; esac
