#!/usr/bin/env bash
# Run on David's PC (WSL). Unattended CLS-0 resource benchmark (Qwen3-8B + Qwen3-0.6B): create an A40 pod on volume 29qhoqghqf (CA-MTL-1), set up a SEPARATE
# checkout of being/v3-bind (the existing Continuous_brain working tree with the A0/Phase-1 results is not touched),
# verify the checkout matches the frozen CLS BENCH commit, launch, wait for completion (2 h cap), download + verify the bundle to ~/cb_results/cls_bench, then remove the pod.
#   bash cb_cls_bench_run.sh [gpu="NVIDIA A40"]
GPU=${1:-NVIDIA A40}; FROZEN=ba093cc; VOL=29qhoqghqf; DC=CA-MTL-1; K=$HOME/.runpod/ssh/runpodctl-ssh-key
case "$FROZEN" in *PLACEHOLDER*) echo "STOP: frozen commit not set"; exit 1;; esac
D=/workspace/Continuous_brain_bind; CAP=$((2 * 3600)); t0=$(date +%s)
runpodctl network-volume get $VOL >/dev/null || { echo "STOP: volume $VOL not found"; exit 1; }
runpodctl pod create --name cb-cls-bench --template-id runpod-torch-v280 --gpu-id "$GPU" --data-center-ids $DC \
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
$S "bash -s" <<REMOTE || { echo "STOP: setup/launch failed - removing pod"; runpodctl pod remove $POD && POD=""; exit 1; }
set -Eeuo pipefail
findmnt -no SOURCE /workspace | grep -q networkvolumes/$VOL
if [ ! -d $D ]; then git clone -q --depth 50 -b being/v3-bind https://github.com/razerblade72611/Continuous_brain $D; fi
cd $D
if pgrep -f '[s]cripts/bind_' >/dev/null; then echo 'STOP: a bind_* process is running in this pod'; exit 1; fi
if ! git diff --quiet || ! git diff --cached --quiet; then echo 'STOP: tracked files modified in the pod checkout'; git status --short; exit 1; fi
# untracked files under results/ (e.g. BIND-D1 outputs now committed upstream) are MOVED aside with their paths, never deleted
B=/workspace/bind_untracked_backup_\$(date -u +%Y%m%dT%H%M%SZ)
git ls-files -z -o --exclude-standard -- results > /tmp/untracked.lst
if [ -s /tmp/untracked.lst ]; then
  mkdir -p "\$B"
  while IFS= read -r -d '' f; do
    if [ -e "\$B/\$f" ]; then echo "STOP: backup collision \$f"; exit 1; fi
    mkdir -p "\$B/\$(dirname "\$f")"
    mv -n "\$f" "\$B/\$f"
    printf '%s\n' "\$f" >> "\$B/MOVED.txt"
  done < /tmp/untracked.lst
  echo "moved \$(wc -l < "\$B/MOVED.txt") untracked result files to \$B"
fi
git fetch -q --depth 50 origin being/v3-bind
git checkout -q -B being/v3-bind FETCH_HEAD
git merge-base --is-ancestor $FROZEN HEAD || { echo 'STOP: checkout is not $FROZEN or a descendant'; exit 1; }
git diff --quiet $FROZEN HEAD -- being scripts/cls_bench.py scripts/launch_cls_bench.sh tests/test_cls_episodic.py \
  || { echo 'STOP: frozen experiment files differ from $FROZEN'; exit 1; }
git log --oneline -1
ln -sfn /workspace/Continuous_brain/.venv .venv
ln -sfn /workspace/Continuous_brain/models models
bash scripts/launch_cls_bench.sh < /dev/null
REMOTE
echo "== waiting for CLS BENCH (checks every 2 min, cap 2 h)"
while :; do
  sleep 120; addr; S="ssh -o ConnectTimeout=10 -i $K -p $PORT root@$IP"
  if $S "grep -q 'CLS BENCH ALL DONE' $D/cls_bench.log" 2>/dev/null; then echo "== done"; break; fi
  echo "$(date +%H:%M) running: $($S "tail -n 1 $D/cls_bench.log | cut -c1-100; for f in $D/results/cls/bench_*.log; do tail -n 1 \$f | cut -c1-100; done" 2>/dev/null | tr '\n' ' ')"
  [ $(( $(date +%s) - t0 )) -gt $CAP ] && { echo "== 2-hour cap reached - stopping pod (results stay on the volume)"; runpodctl pod stop $POD; POD=""; exit 1; }
done
$S "cat $D/cls_bench.log"
echo "== benchmark results are in the log above and in the bundle (results/cls/bench_*.json)"
mkdir -p ~/cb_results/cls_bench
$S "cd $D && tar cf - results_cls_bench.tar.gz results_cls_bench.tar.gz.sha256 cls_bench.log" | tar xf - -C ~/cb_results/cls_bench \
  && (cd ~/cb_results/cls_bench && sha256sum -c results_cls_bench.tar.gz.sha256) \
  && cp ~/cb_results/cls_bench/results_cls_bench.tar.gz /mnt/c/Users/davei/Downloads/ \
  && echo "== downloaded + verified: ~/cb_results/cls_bench (copy in Downloads)" \
  && runpodctl pod remove $POD >/dev/null && echo "== pod removed (results also stay on the volume)" && DONE=1
[ "$DONE" = 1 ] || { echo "== download/verification did not complete - pod stopped by cleanup, files stay on the volume"; exit 1; }
# propagate the benchmark's own status (a verified bundle is not success): COMPLETE 0, PARTIAL 2, FAILED 1
ST=$(grep -o 'status=BENCH_[A-Z]*' ~/cb_results/cls_bench/cls_bench.log | tail -n 1 | cut -d= -f2)
echo "== benchmark status: ${ST:-UNKNOWN}"
case "$ST" in BENCH_COMPLETE) exit 0;; BENCH_PARTIAL) exit 2;; *) exit 1;; esac
