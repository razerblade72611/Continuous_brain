#!/usr/bin/env bash
# WSL: every 5 min copy the r3t pod's log + status home; when it finishes, download the results, verify sha256,
# and only then DELETE the pod.  Usage: ~/cb2_sync_r3t.sh <pod-id> <ip> <port>
POD=$1; IP=$2; P=$3; KEY=/home/davei/.runpod/ssh/runpodctl-ssh-key; H=root@$IP
OUT=/mnt/c/Users/davei/ContinuousBrain/cb_exchange
SSH="ssh -i $KEY -p $P -o ConnectTimeout=20 -o BatchMode=yes $H"
while true; do
  $SSH 'date -u; ps -eo etime,cmd | grep -E "^ *[0-9:-]+ +(bash /workspace|bash scripts|python)" | cut -c1-150; ls /workspace/Continuous_brain/runs/explore/cb2/logs/r3t/ 2>/dev/null; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; free -g | head -2; uptime' > "$OUT/r3t_status.txt" 2>&1
  scp -q -i $KEY -P $P $H:/workspace/r3t.log "$OUT/r3t_pod.log"
  for f in practice train eval-none eval-branch eval-full; do
    scp -q -i $KEY -P $P $H:/workspace/Continuous_brain/runs/explore/cb2/logs/r3t/$f.log "$OUT/r3t_$f.log" 2>/dev/null
  done
  if grep -q 'r3t pod finished' "$OUT/r3t_pod.log" 2>/dev/null; then
    scp -q -i $KEY -P $P $H:/workspace/cb2_r3t_results.tgz "$OUT/" && \
    R=$($SSH 'sha256sum /workspace/cb2_r3t_results.tgz' | cut -d' ' -f1) && L=$(sha256sum "$OUT/cb2_r3t_results.tgz" | cut -d' ' -f1) && \
    [ -n "$R" ] && [ "$R" = "$L" ] && { echo "cb2_r3t_results.tgz DOWNLOADED, sha256 MATCH $L at $(date)" >> "$OUT/r3t_status.txt"; \
      cd ~/projects/continuous_brain_explore && ./runpodctl pod delete "$POD" >> "$OUT/r3t_status.txt" 2>&1; exit 0; }
  fi
  if ! grep -q 'bash /workspace/cb2_r3t_pod.sh' "$OUT/r3t_status.txt" && ! grep -q 'r3t pod finished' "$OUT/r3t_pod.log" 2>/dev/null \
     && grep -q 'STOP\|Error\|Traceback' "$OUT/r3t_pod.log" 2>/dev/null; then
    echo "POD RUN FAILED (pod kept for debugging; it is still billing): see r3t_pod.log" >> "$OUT/r3t_status.txt"
  fi
  sleep 300
done
