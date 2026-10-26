#!/usr/bin/env bash
# Billing watchdog, run on David's PC (WSL): stops the pod 10 min after the stage finishes (if it hasn't stopped itself)
# or at a hard time cap. Stopping keeps everything on the network volume.
#   bash cb_watchdog.sh <pod-id> [log=stageA1run2.log] [cap_hours=7]
POD=${1:?usage: cb_watchdog.sh <pod-id> [log] [cap_hours]}; LOG=${2:-stageA1run2.log}; CAP=${3:-7}
KEY=${KEY:-$HOME/.runpod/ssh/runpodctl-ssh-key}; start=$(date +%s); done_at=""
echo "watchdog: pod $POD, log $LOG, cap ${CAP}h"
while true; do
  st=$(runpodctl pod get "$POD" 2>/dev/null | python3 -c 'import json,sys
try: print(json.load(sys.stdin).get("desiredStatus"))
except Exception: print("GONE")')
  [ "$st" != "RUNNING" ] && { echo "$(date +%H:%M) pod is $st - watchdog done"; break; }
  eval "$(runpodctl ssh info "$POD" 2>/dev/null | python3 -c 'import sys,json
try: d=json.load(sys.stdin); print("IP=%s PORT=%s" % (d["ip"], d["port"]))
except Exception: pass')"
  if [ -z "$done_at" ] && [ -n "$IP" ] && ssh -o ConnectTimeout=10 -i "$KEY" -p "$PORT" "root@$IP" \
       "grep -qE 'STAGE A1 DONE|STAGE A DONE|A0 VERDICT|STAGE STOPPED|BIND-D1 ALL DONE' /workspace/Continuous_brain/$LOG 2>/dev/null"; then
    done_at=$(date +%s); echo "$(date +%H:%M) stage finished - giving the pod 10 min to stop itself"
  fi
  if [ -n "$done_at" ] && [ $(( $(date +%s) - done_at )) -gt 600 ]; then
    echo "$(date +%H:%M) still running - stopping pod"; runpodctl pod stop "$POD"
  fi
  if [ $(( $(date +%s) - start )) -gt $(( CAP * 3600 )) ]; then
    echo "$(date +%H:%M) ${CAP}-hour cap - stopping pod"; runpodctl pod stop "$POD"
  fi
  echo "$(date +%H:%M) pod $st, stage $([ -n "$done_at" ] && echo finished || echo running)"
  sleep 300
done
