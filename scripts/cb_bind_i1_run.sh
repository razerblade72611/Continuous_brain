#!/usr/bin/env bash
# Run on David's PC (WSL). Unattended BIND-I1 (seeds 2-5 in parallel): create an A40 pod on volume 29qhoqghqf (CA-MTL-1), set up a SEPARATE
# checkout of being/v3-bind (the existing Continuous_brain working tree with the A0/Phase-1 results is not touched),
# launch, wait for completion (3 h cap), download + verify the bundle to ~/cb_results/bind_i1, then remove the pod.
#   bash cb_bind_i1_run.sh [gpu="NVIDIA A40"]
GPU=${1:-NVIDIA A40}; VOL=29qhoqghqf; DC=CA-MTL-1; K=$HOME/.runpod/ssh/runpodctl-ssh-key
D=/workspace/Continuous_brain_bind; CAP=$((3 * 3600)); t0=$(date +%s)
runpodctl network-volume get $VOL >/dev/null || { echo "STOP: volume $VOL not found"; exit 1; }
runpodctl pod create --name cb-bind-i1 --template-id runpod-torch-v280 --gpu-id "$GPU" --data-center-ids $DC \
  --network-volume-id $VOL --volume-mount-path /workspace --ssh --wait --wait-timeout 20m > ~/cb_bind_pod.json \
  || { echo "STOP: pod create failed"; cat ~/cb_bind_pod.json; exit 1; }
POD=$(python3 -c 'import json;print(json.load(open("'"$HOME"'/cb_bind_pod.json"))["id"])') || exit 1
echo "== pod $POD ($GPU, $DC)"
addr() { eval "$(runpodctl ssh info $POD 2>/dev/null | python3 -c 'import sys,json
try: d=json.load(sys.stdin); print("IP=%s PORT=%s" % (d["ip"], d["port"]))
except Exception: pass')"; }
addr; S="ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 -i $K -p $PORT root@$IP"
$S "set -e; findmnt -no SOURCE /workspace | grep -q networkvolumes/$VOL
  [ -d $D ] || git clone -q --depth 5 -b being/v3-bind https://github.com/razerblade72611/Continuous_brain $D
  cd $D && git fetch -q origin being/v3-bind && git checkout -q -B being/v3-bind FETCH_HEAD && git log --oneline -1
  ln -sfn /workspace/Continuous_brain/.venv .venv; ln -sfn /workspace/Continuous_brain/models models
  bash scripts/launch_bind_i1.sh" || { echo "STOP: setup/launch failed - removing pod"; runpodctl pod remove $POD; exit 1; }
echo "== waiting for BIND-I1 (checks every 2 min, cap 3 h)"
while :; do
  sleep 120; addr; S="ssh -o ConnectTimeout=10 -i $K -p $PORT root@$IP"
  if $S "grep -q 'BIND-I1 ALL DONE' $D/bind_i1.log" 2>/dev/null; then echo "== done"; break; fi
  echo "$(date +%H:%M) running: $($S "for s in 2 3 4 5; do tail -n 1 $D/results/dev3/bind_i1_seed\$s.log | cut -c1-60; done" 2>/dev/null | tr '\n' ' ')"
  [ $(( $(date +%s) - t0 )) -gt $CAP ] && { echo "== 3-hour cap reached - stopping pod (results stay on the volume)"; runpodctl pod stop $POD; exit 1; }
done
$S "cat $D/bind_i1.log"
echo "== analysis is in the log above and in the bundle (results/dev3/bind_i1_analysis.txt)"
mkdir -p ~/cb_results/bind_i1
$S "cd $D && tar cf - results_bind_i1.tar.gz results_bind_i1.tar.gz.sha256 bind_i1.log" | tar xf - -C ~/cb_results/bind_i1 \
  && (cd ~/cb_results/bind_i1 && sha256sum -c results_bind_i1.tar.gz.sha256) \
  && cp ~/cb_results/bind_i1/results_bind_i1.tar.gz /mnt/c/Users/davei/Downloads/ \
  && echo "== downloaded + verified: ~/cb_results/bind_i1 (copy in Downloads)" \
  && runpodctl pod remove $POD >/dev/null && echo "== pod removed (results also stay on the volume)"
