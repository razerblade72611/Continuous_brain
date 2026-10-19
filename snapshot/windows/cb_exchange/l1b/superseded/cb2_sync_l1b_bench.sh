#!/usr/bin/env bash
# WSL: watch the L1b bench pod every 2 min (status + log into cb_exchange/l1b).  When the bench finishes: download
# cb2_l1b_bench.tgz, verify sha256, then DELETE the pod.  If the bench fails, or runs past MAXH hours: save the logs
# and DELETE the pod anyway (a bench is cheap to redo; a forgotten pod is not -- runbook 6).
#   ~/cb2_sync_l1b_bench.sh <pod-id> <ip> <port>
POD=$1; IP=$2; P=$3; KEY=/home/davei/.runpod/ssh/runpodctl-ssh-key; H=root@$IP; MAXH=${MAXH:-3}
OUT=/mnt/c/Users/davei/ContinuousBrain/cb_exchange/l1b; ST=$OUT/bench_status.txt
SSH="ssh -i $KEY -p $P -o ConnectTimeout=20 -o BatchMode=yes $H"
RP=~/projects/continuous_brain_explore/runpodctl
T0=$(date +%s)
finish() {   # $1 = reason; logs first, then delete, then record
  scp -q -i $KEY -P $P "$H:/workspace/l1b-*.json" "$H:/workspace/l1b-rehearsal.log" "$OUT/" 2>/dev/null
  echo "$(date -u) $1 -- deleting pod $POD" >> "$ST"
  $RP pod delete "$POD" >> "$ST" 2>&1 && echo "$(date -u) pod $POD deleted" >> "$ST"
  exit 0
}
while true; do
  $SSH 'date -u; ps -eo etime,cmd | grep -E "^ *[0-9:-]+ +(bash /workspace|python)" | cut -c1-150 | head -6; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; free -g | head -2' > "$ST.tmp" 2>&1 && mv "$ST.tmp" "$ST"
  scp -q -i $KEY -P $P $H:/workspace/l1b-bench.log "$OUT/bench_pod.log"
  if grep -q 'l1b BENCH finished' "$OUT/bench_pod.log" 2>/dev/null; then
    scp -q -i $KEY -P $P $H:/workspace/cb2_l1b_bench.tgz "$OUT/" && \
    R=$($SSH 'sha256sum /workspace/cb2_l1b_bench.tgz' | cut -d' ' -f1) && Lh=$(sha256sum "$OUT/cb2_l1b_bench.tgz" | cut -d' ' -f1) && \
    [ -n "$R" ] && [ "$R" = "$Lh" ] && { echo "$(date -u) cb2_l1b_bench.tgz DOWNLOADED, sha256 MATCH $Lh" >> "$ST"; finish "bench finished"; }
  fi
  # (the watcher starts before the bench does: judge "not running" only after a 10-minute grace period)
  if [ $(( $(date +%s) - T0 )) -gt 600 ] && ! grep -q 'bash /workspace/cb2_l1b_pod.sh' "$ST" \
     && ! grep -q 'l1b BENCH finished' "$OUT/bench_pod.log" 2>/dev/null; then
    sleep 30; scp -q -i $KEY -P $P $H:/workspace/l1b-bench.log "$OUT/bench_pod.log"
    grep -q 'l1b BENCH finished' "$OUT/bench_pod.log" 2>/dev/null || finish "BENCH FAILED or stopped (see bench_pod.log)"
  fi
  [ $(( $(date +%s) - T0 )) -gt $(( MAXH * 3600 )) ] && finish "bench exceeded ${MAXH} h"
  sleep 120
done
