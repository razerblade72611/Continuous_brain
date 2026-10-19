#!/usr/bin/env bash
# WSL: every 5 min copy the S2c pod's log + status home; when it finishes, download the results, verify sha256,
# and only then DELETE the pod.  Usage: ~/cb2_sync_s2c.sh <pod-id> <ip> <port>
POD=$1; IP=$2; P=$3; KEY=/home/davei/.runpod/ssh/runpodctl-ssh-key; H=root@$IP
OUT=/mnt/c/Users/davei/ContinuousBrain/cb_exchange
SSH="ssh -i $KEY -p $P -o ConnectTimeout=20 -o BatchMode=yes $H"
while true; do
  $SSH 'date -u; ps -eo etime,cmd | grep -E "^ *[0-9:-]+ +(bash /workspace|bash scripts|python)" | cut -c1-150 | head -8; ps -e | grep -c python; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; free -g | head -2; uptime' > "$OUT/s2c_status.txt" 2>&1
  scp -q -i $KEY -P $P $H:/workspace/s2c.log "$OUT/s2c_pod.log"
  for f in train sealed install; do
    scp -q -i $KEY -P $P $H:/workspace/Continuous_brain/runs/explore/cb2/logs/s2c/$f.log "$OUT/s2c_$f.log" 2>/dev/null
  done
  scp -q -i $KEY -P $P $H:/workspace/Continuous_brain/runs/explore/cb2/s2c/r6/dev.json "$OUT/s2c_dev.json" 2>/dev/null
  if grep -q 's2c pod finished' "$OUT/s2c_pod.log" 2>/dev/null; then
    scp -q -i $KEY -P $P $H:/workspace/cb2_s2c_results.tgz "$OUT/" && \
    R=$($SSH 'sha256sum /workspace/cb2_s2c_results.tgz' | cut -d' ' -f1) && L=$(sha256sum "$OUT/cb2_s2c_results.tgz" | cut -d' ' -f1) && \
    [ -n "$R" ] && [ "$R" = "$L" ] && { echo "cb2_s2c_results.tgz DOWNLOADED, sha256 MATCH $L at $(date)" >> "$OUT/s2c_status.txt"; \
      cd ~/projects/continuous_brain_explore && ./runpodctl pod delete "$POD" >> "$OUT/s2c_status.txt" 2>&1; exit 0; }
  fi
  if ! grep -q 'bash /workspace/cb2_s2c_pod.sh' "$OUT/s2c_status.txt" && ! grep -q 's2c pod finished' "$OUT/s2c_pod.log" 2>/dev/null \
     && grep -q 'STOP\|Error\|Traceback' "$OUT/s2c_pod.log" 2>/dev/null; then
    echo "POD RUN FAILED (pod kept for debugging; it is still billing): see s2c_pod.log" >> "$OUT/s2c_status.txt"
  fi
  sleep 300
done
