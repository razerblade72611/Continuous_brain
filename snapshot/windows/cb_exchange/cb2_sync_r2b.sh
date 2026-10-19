#!/usr/bin/env bash
# Every 5 min: copy the r2 resume logs + status home. When results exist: download, verify sha256, then
# download the evaluation/dream archive too, verify both, then DELETE the pod (only after both hashes match) and exit.
# Usage (WSL): ~/cb2_sync_r2b.sh <pod-id> <ip> <port>
POD=$1; IP=$2; P=$3; KEY=/home/davei/.runpod/ssh/runpodctl-ssh-key; H=root@$IP
OUT=/mnt/c/Users/davei/ContinuousBrain/cb_exchange
SSH="ssh -i $KEY -p $P -o ConnectTimeout=20 -o BatchMode=yes $H"
while true; do
  $SSH 'date -u; ps -eo etime,cmd | grep -E "^ *[0-9:-]+ +(bash /workspace/|python)" | cut -c1-140; ls /workspace/Continuous_brain/runs/explore/cb2/logs/r2/ 2>/dev/null; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; uptime' > "$OUT/r2b_status.txt" 2>&1
  scp -q -i $KEY -P $P $H:/workspace/r2_resume.log "$OUT/r2b_resume.log"
  scp -q -i $KEY -P $P $H:/workspace/r2.log "$OUT/r2b_pod.log" 2>/dev/null
  if grep -q 'r2 resume finished' "$OUT/r2b_resume.log" 2>/dev/null; then
    ok=1; for f in cb2_results_r2.tgz cb2_r2_archive_extra.tgz; do
      scp -q -i $KEY -P $P $H:/workspace/$f "$OUT/" || ok=0
      R=$($SSH "sha256sum /workspace/$f" | cut -d' ' -f1); L=$(sha256sum "$OUT/$f" 2>/dev/null | cut -d' ' -f1)
      [ -n "$R" ] && [ "$R" = "$L" ] && echo "$f DOWNLOADED, sha256 MATCH $L" >> "$OUT/r2b_status.txt" || ok=0
    done
    [ $ok = 1 ] && { echo "ALL RESULTS VERIFIED at $(date)" >> "$OUT/r2b_status.txt"; cd ~/projects/continuous_brain_explore && ./runpodctl pod delete "$POD" >> "$OUT/r2b_status.txt" 2>&1; exit 0; }
  fi
  sleep 300
done
