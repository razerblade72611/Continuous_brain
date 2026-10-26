#!/usr/bin/env bash
# Run on David's PC (WSL). Shows the pod workspace status; if the stage has finished, downloads its bundle, summary and
# log to ~/cb_results/<name>, verifies the checksum, then stops the pod. Starts a stopped pod only to read.
#   bash cb_fetch.sh <pod-id> [name=stageA2_A1run2] [log=stageA1run2.log]
POD=${1:?usage: cb_fetch.sh <pod-id> [name] [log]}; NAME=${2:-stageA2_A1run2}; LOG=${3:-stageA1run2.log}
K=${KEY:-$HOME/.runpod/ssh/runpodctl-ssh-key}; R=/workspace/Continuous_brain
st=$(runpodctl pod get "$POD" 2>/dev/null | python3 -c 'import json,sys
try: print(json.load(sys.stdin).get("desiredStatus"))
except Exception: print("GONE")')
echo "== pod $POD: $st"
[ "$st" = "GONE" ] && { echo "pod no longer exists - create a new pod on volume 29qhoqghqf to read the results"; exit 1; }
[ "$st" = "RUNNING" ] || { runpodctl pod start "$POD" >/dev/null && echo "== started the pod to read results" || { echo "pod start failed"; exit 1; }; }
IP=""
for i in $(seq 1 30); do
  eval "$(runpodctl ssh info "$POD" 2>/dev/null | python3 -c 'import sys,json
try: d=json.load(sys.stdin); print("IP=%s PORT=%s" % (d["ip"], d["port"]))
except Exception: pass')"
  [ -n "$IP" ] && ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new -i "$K" -p "$PORT" "root@$IP" true 2>/dev/null && break
  IP=""; sleep 10
done
[ -z "$IP" ] && { echo "could not reach the pod over ssh"; exit 1; }
S="ssh -i $K -p $PORT root@$IP"
$S "cd $R && git pull -q; bash scripts/cb_status.sh" 2>&1 | tee ~/cb_check.txt
if $S "test -f $R/results_${NAME}.tar.gz.sha256 && ! pgrep -f '[s]cripts/(stage_v3|bind_d1)'"; then
  mkdir -p ~/cb_results/"$NAME"
  $S "cd $R && tar cf - results_${NAME}.tar.gz results_${NAME}.tar.gz.sha256 $LOG results/dev3/${NAME}_summary.json" \
    | tar xf - -C ~/cb_results/"$NAME" \
    && (cd ~/cb_results/"$NAME" && sha256sum -c "results_${NAME}.tar.gz.sha256") \
    && echo "== downloaded and verified: ~/cb_results/$NAME" \
    && runpodctl pod stop "$POD" >/dev/null && echo "== pod stopped (everything stays on the network volume)"
else
  echo "== stage still running (or no summary yet) - pod left running"
fi
