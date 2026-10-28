#!/usr/bin/env bash
# Run on David's PC (WSL). Unattended CLS-0B (default: pilot seeds 2 and 3, one worker each): create an A40 pod on volume 29qhoqghqf (CA-MTL-1), set up a SEPARATE
# checkout of being/v3-bind (the existing Continuous_brain working tree with the A0/Phase-1 results is not touched),
# verify the checkout matches the frozen CLS-0B commit, launch, wait for completion (8 h cap), download + verify the bundle to ~/cb_results/cls0b, then remove the pod.
#   bash cb_cls0b_run.sh [seeds="2 3"] [new|resume] [gpu="NVIDIA A40"]
SEEDS=${1:-2 3}; MODE=${2:-new}; GPU=${3:-NVIDIA A40}; FROZEN=39a4ccb; VOL=29qhoqghqf; DC=CA-MTL-1; K=$HOME/.runpod/ssh/runpodctl-ssh-key
case "$FROZEN" in *PLACEHOLDER*) echo "STOP: frozen commit not set"; exit 1;; esac
D=/workspace/Continuous_brain_bind; CAP=$((8 * 3600)); t0=$(date +%s)
runpodctl network-volume get $VOL >/dev/null || { echo "STOP: volume $VOL not found"; exit 1; }
runpodctl pod create --name cb-cls0b --template-id runpod-torch-v280 --gpu-id "$GPU" --data-center-ids $DC \
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
# CLS-0B banks and progress files are resumable checkpoints: they stay in place (never moved aside)
git ls-files -z -o --exclude-standard -- results ':(exclude)results/cls0b' > /tmp/untracked.lst
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
git diff --quiet $FROZEN HEAD -- being scripts/bind_i1.py scripts/cls0b.py scripts/cls0b_analyze.py scripts/launch_cls0b.sh tests/test_cls0b.py lab/CLS0B_PROTOCOL.md results/dev3/bind_i1 \
  || { echo 'STOP: frozen experiment files differ from $FROZEN'; exit 1; }
git log --oneline -1
ln -sfn /workspace/Continuous_brain/.venv .venv
ln -sfn /workspace/Continuous_brain/models models
bash scripts/launch_cls0b.sh "$SEEDS" "$MODE" < /dev/null
REMOTE
echo "== waiting for CLS-0B (checks every 2 min, cap 8 h)"
while :; do
  sleep 120; addr; S="ssh -o ConnectTimeout=10 -i $K -p $PORT root@$IP"
  if $S "grep -q 'CLS-0B ALL DONE' $D/cls0b.log" 2>/dev/null; then echo "== done"; break; fi
  echo "$(date +%H:%M) running: $($S "for f in $D/results/cls0b/seed*.log; do tail -n 1 \$f | cut -c1-90; done" 2>/dev/null | tr '\n' ' ')"
  [ $(( $(date +%s) - t0 )) -gt $CAP ] && { echo "== 8-hour cap reached - stopping pod (results stay on the volume)"; runpodctl pod stop $POD; POD=""; exit 1; }
done
$S "cat $D/cls0b.log"
echo "== results and the frozen analysis are in the log above and in the bundle (results/cls0b/)"
mkdir -p ~/cb_results/cls0b
$S "cd $D && tar cf - results_cls0b.tar.gz results_cls0b.tar.gz.sha256 cls0b.log" | tar xf - -C ~/cb_results/cls0b \
  && (cd ~/cb_results/cls0b && sha256sum -c results_cls0b.tar.gz.sha256) \
  && cp ~/cb_results/cls0b/results_cls0b.tar.gz /mnt/c/Users/davei/Downloads/ \
  && echo "== downloaded + verified: ~/cb_results/cls0b (copy in Downloads)" \
  && runpodctl pod remove $POD >/dev/null && echo "== pod removed (results also stay on the volume)" && DONE=1
[ "$DONE" = 1 ] || { echo "== download/verification did not complete - pod stopped by cleanup, files stay on the volume"; exit 1; }
# propagate the run's own status (a verified bundle is not success): COMPLETE 0, otherwise 1
ST=$(grep -o 'status=CLS0B_[A-Z_]*' ~/cb_results/cls0b/cls0b.log | tail -n 1 | cut -d= -f2)
echo "== CLS-0B status: ${ST:-UNKNOWN}"
case "$ST" in CLS0B_COMPLETE) exit 0;; CLS0B_COLLECTED_INVALID) exit 3;; CLS0B_PREFLIGHT_STOP) exit 4;; *) exit 1;; esac
