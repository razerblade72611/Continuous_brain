#!/usr/bin/env bash
# WSL: every 5 min copy the r3d pod's log + status home; when it finishes, download the results, verify sha256,
# and only then DELETE the pod.  Usage: ~/cb2_sync_r3d.sh <pod-id> <ip> <port>
POD=$1; IP=$2; P=$3; KEY=/home/davei/.runpod/ssh/runpodctl-ssh-key; H=root@$IP
OUT=/mnt/c/Users/davei/ContinuousBrain/cb_exchange
SSH="ssh -i $KEY -p $P -o ConnectTimeout=20 -o BatchMode=yes $H"
while true; do
  $SSH 'date -u; ps -eo etime,cmd | grep -E "^ *[0-9:-]+ +(bash /workspace|bash scripts|python)" | cut -c1-150; ls /workspace/Continuous_brain/runs/explore/cb2/logs/r3d/ 2>/dev/null; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; uptime' > "$OUT/r3d_status.txt" 2>&1
  scp -q -i $KEY -P $P $H:/workspace/r3d.log "$OUT/r3d_pod.log"
  for f in control train-i1 train-i2 train-i3 eval-dagger eval-control; do
    scp -q -i $KEY -P $P $H:/workspace/Continuous_brain/runs/explore/cb2/logs/r3d/$f.log "$OUT/r3d_$f.log" 2>/dev/null
  done
  if grep -q 'r3d pod finished' "$OUT/r3d_pod.log" 2>/dev/null; then
    scp -q -i $KEY -P $P $H:/workspace/cb2_r3d_results.tgz "$OUT/" && \
    R=$($SSH 'sha256sum /workspace/cb2_r3d_results.tgz' | cut -d' ' -f1) && L=$(sha256sum "$OUT/cb2_r3d_results.tgz" | cut -d' ' -f1) && \
    [ -n "$R" ] && [ "$R" = "$L" ] && { echo "cb2_r3d_results.tgz DOWNLOADED, sha256 MATCH $L at $(date)" >> "$OUT/r3d_status.txt"; \
      cd ~/projects/continuous_brain_explore && ./runpodctl pod delete "$POD" >> "$OUT/r3d_status.txt" 2>&1; exit 0; }
  fi
  sleep 300
done
