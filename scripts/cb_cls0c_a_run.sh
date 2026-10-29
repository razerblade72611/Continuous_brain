#!/usr/bin/env bash
# Run on David's PC (WSL). CLS-0C Phase A (read-only diagnostic, lab/CLS0C_A_PROTOCOL.md): create an A40 pod on volume
# 29qhoqghqf (CA-MTL-1), use a SEPARATE checkout pinned to the diagnostic commit (the CLS-0B checkout and its results on
# the volume are not touched), verify the committed CLS-0B inputs, run the diagnostic (45 min cap), download + verify the
# bundle to ~/cb_results/cls0c_a, then remove the pod. Exit: 0 valid, 3 invalid (no conclusion), 1 operational failure.
#   bash cb_cls0c_a_run.sh [gpu="NVIDIA A40"]
GPU=${1:-NVIDIA A40}; VOL=29qhoqghqf; DC=CA-MTL-1; K=$HOME/.runpod/ssh/runpodctl-ssh-key
PIN=f416712bfa981b5e6b025de0582587939a747913                   # diagnostic commit (scripts/cls0c_a.py)
D=/workspace/Continuous_brain_c0ca; CAP=$((45 * 60)); t0=$(date +%s)
runpodctl network-volume get $VOL >/dev/null || { echo "STOP: volume $VOL not found"; exit 1; }
runpodctl pod create --name cb-cls0c-a --template-id runpod-torch-v280 --gpu-id "$GPU" --data-center-ids $DC \
  --network-volume-id $VOL --volume-mount-path /workspace --ssh --wait --wait-timeout 20m > ~/cb_c0ca_pod.json \
  || { echo "STOP: pod create failed"; cat ~/cb_c0ca_pod.json; exit 1; }
POD=$(python3 -c 'import json;print(json.load(open("'"$HOME"'/cb_c0ca_pod.json"))["id"])') || exit 1
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
if ! git diff --quiet || ! git diff --cached --quiet; then echo 'STOP: tracked files modified in the diagnostic checkout'; git status --short; exit 1; fi
if [ -e results/cls0c_a ]; then mv results/cls0c_a results/cls0c_a.$(date -u +%Y%m%dT%H%M%SZ); echo 'earlier diagnostic output moved aside'; fi
git fetch -q --depth 50 origin being/v3-bind
git cat-file -e $PIN^{commit}
git checkout -q --detach $PIN
[ "$(git rev-parse HEAD)" = "$PIN" ] || { echo 'STOP: HEAD is not the pinned diagnostic commit'; exit 1; }
echo "diagnostic code at: $(git log --oneline -1)"
(cd results/cls0b_final && sha256sum -c --quiet FILES.sha256) || { echo 'STOP: CLS-0B inputs do not match FILES.sha256'; exit 1; }
ln -sfn /workspace/Continuous_brain/.venv .venv
[ -f /workspace/hf/Qwen3-Embedding-0.6B/config.json ] || { echo 'STOP: embedder missing'; exit 1; }
rm -f /workspace/cls0c_a.log /workspace/results_cls0c_a.tar.gz /workspace/results_cls0c_a.tar.gz.sha256
cat > /workspace/cls0c_a_job.sh <<'JOB'
cd /workspace/Continuous_brain_c0ca
timeout 2400 .venv/bin/python scripts/cls0c_a.py --results results/cls0b_final --external /workspace/hf/Qwen3-Embedding-0.6B \
    --out results/cls0c_a; e=$?
echo "diagnostic exit $e"
tar czf /workspace/results_cls0c_a.tar.gz results/cls0c_a
sha256sum /workspace/results_cls0c_a.tar.gz | sed 's|/workspace/||' > /workspace/results_cls0c_a.tar.gz.sha256 && sync
echo "BUNDLE $(cat /workspace/results_cls0c_a.tar.gz.sha256)"
echo "CLS-0C-A JOB DONE exit=$e"
JOB
nohup bash /workspace/cls0c_a_job.sh > /workspace/cls0c_a.log 2>&1 < /dev/null &
sleep 5; echo "DIAGNOSTIC LAUNCHED"
REMOTE
echo "== waiting for the diagnostic (checks every minute, cap 45 min)"
while :; do
  sleep 60; addr; S="ssh -o ConnectTimeout=10 -i $K -p $PORT root@$IP"
  if $S "grep -q 'CLS-0C-A JOB DONE' /workspace/cls0c_a.log" 2>/dev/null; then echo "== done"; break; fi
  echo "$(date +%H:%M) running: $($S "tail -n 1 /workspace/cls0c_a.log | cut -c1-100" 2>/dev/null)"
  [ $(( $(date +%s) - t0 )) -gt $CAP ] && { echo "== 45-min cap reached - stopping pod"; runpodctl pod stop $POD; POD=""; exit 1; }
done
$S "grep -v 'Loading weights' /workspace/cls0c_a.log"
mkdir -p ~/cb_results/cls0c_a
$S "cd /workspace && tar cf - results_cls0c_a.tar.gz results_cls0c_a.tar.gz.sha256 cls0c_a.log" | tar xf - -C ~/cb_results/cls0c_a \
  && (cd ~/cb_results/cls0c_a && sha256sum -c results_cls0c_a.tar.gz.sha256) \
  && cp ~/cb_results/cls0c_a/results_cls0c_a.tar.gz /mnt/c/Users/davei/Downloads/ \
  && echo "== downloaded + verified: ~/cb_results/cls0c_a (copy in Downloads)" \
  && runpodctl pod remove $POD >/dev/null && echo "== pod removed" && DONE=1
[ "$DONE" = 1 ] || { echo "== download/verification did not complete - pod stopped by cleanup, files stay on the volume"; exit 1; }
E=$(grep -o 'CLS-0C-A JOB DONE exit=[0-9]*' ~/cb_results/cls0c_a/cls0c_a.log | tail -n 1 | cut -d= -f2)
echo "== CLS-0C-A diagnostic exit: ${E:-UNKNOWN}"
case "$E" in 0) exit 0;; 3) exit 3;; *) exit 1;; esac
