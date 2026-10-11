#!/usr/bin/env bash
# WSL: CB2-L1b PHASE=bench on one RTX 4090 (~1 h): GPU rehearsal + bench-sleep + bench-lives, then the watcher
# downloads cb2_l1b_bench.tgz, verifies it, and DELETES the pod.  The loop (PHASE=run) is NOT started by this.
#   bash /mnt/c/Users/davei/ContinuousBrain/cb_exchange/l1b/cb2_launch_l1b_bench.sh             # create pod + start
#   bash .../cb2_launch_l1b_bench.sh <pod-id> <ip> <ssh-port>                                   # use an existing pod
# CLOUD=COMMUNITY (default, ~$0.34/h) or SECURE (~$0.89/h).  Pod name: cb2-l1b-bench-claude (COORDINATION.md rule 2).
set -euo pipefail
EX=/mnt/c/Users/davei/ContinuousBrain/cb_exchange; L=$EX/l1b
KEY=/home/davei/.runpod/ssh/runpodctl-ssh-key; RP=~/projects/continuous_brain_explore/runpodctl
GPU=${GPU:-"NVIDIA GeForce RTX 4090"}; CLOUD=${CLOUD:-COMMUNITY}; NAME=cb2-l1b-bench-claude
cd "$L" && sha256sum -c cb2_l1b_SHA256SUMS
cd "$EX" && sha256sum -c <<SUMS
0c46b2d1ace00c027d45e23ae01b00da8b802d10660122da49f9727a1cfe9999  tile_decoder_v1.pt
2d3ec50ce05e635a46fc2d658a2395aa65de929088c17879090e751f15a1a1ec  cb2_kit_r2.tgz
81e31395144364bcc55b4f57ac97cb8cdc94947e5c22730a4e584d6b8603c5f8  s2bres/s2b_min.tgz
SUMS
echo "== pods on the account now (runbook 6: delete nothing that is not ours; L1's gjfpfnsfymn8yf is Sol's)"
$RP pod list || true
if [ $# -ge 3 ]; then POD=$1; IP=$2; PORT=$3
else
  echo "== creating $NAME (1 x $GPU, $CLOUD cloud)"
  EXTRA=""; [ "$CLOUD" = "COMMUNITY" ] && EXTRA="--public-ip"
  $RP pod create --name $NAME --gpu-id "$GPU" --gpu-count 1 --cloud-type "$CLOUD" $EXTRA \
      --image "runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404" --container-disk-in-gb 40 --volume-in-gb 40 --ports "22/tcp" \
      | tee /tmp/cb2_l1b_create.txt
  POD=$(grep -oE '"?id"?[": ]+[a-z0-9]{10,16}' /tmp/cb2_l1b_create.txt | head -1 | grep -oE '[a-z0-9]{10,16}$' || true)
  [ -n "$POD" ] || { echo "No pod id above (no capacity on $CLOUD? try CLOUD=SECURE). Nothing to clean up if none was created."; exit 1; }
  echo "pod: $POD -- waiting for SSH"
  for i in $(seq 1 60); do
    $RP pod get "$POD" > /tmp/cb2_l1b_get.txt 2>&1 || true
    read -r IP PORT < <(python3 - /tmp/cb2_l1b_get.txt <<'PY'
import json, re, sys
t = open(sys.argv[1]).read()
ip = port = None
try:
    d = json.loads(t)
    d = d[0] if isinstance(d, list) else d
    ip = d.get("publicIp") or (d.get("runtime") or {}).get("publicIp")
    pm = d.get("portMappings") or {}
    port = pm.get("22") or pm.get(22)
    for p in ((d.get("runtime") or {}).get("ports") or []):
        if p.get("privatePort") == 22 and p.get("isIpPublic"):
            ip, port = p.get("ip"), p.get("publicPort")
except Exception:
    pass
if not (ip and port):
    m = re.search(r"root@(\d+\.\d+\.\d+\.\d+)\s+-p\s+(\d+)", t) or re.search(r"(\d+\.\d+\.\d+\.\d+):(\d+)\s*->\s*22", t)
    if m: ip, port = m.group(1), m.group(2)
print(ip or "", port or "")
PY
)
    [ -n "${IP:-}" ] && [ -n "${PORT:-}" ] && break
    sleep 10
  done
  [ -n "${IP:-}" ] && [ -n "${PORT:-}" ] || { cat /tmp/cb2_l1b_get.txt; echo "SSH not found. Pod $POD EXISTS AND BILLS: re-run with <pod-id> <ip> <port>, or: $RP pod delete $POD"; exit 1; }
fi
echo "pod $POD  ssh root@$IP -p $PORT"
SSH="ssh -i $KEY -p $PORT -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 root@$IP"
for i in $(seq 1 30); do $SSH true 2>/dev/null && break; sleep 10; done
echo "== uploading"
scp -i $KEY -P $PORT -o StrictHostKeyChecking=accept-new "$EX/tile_decoder_v1.pt" "$EX/cb2_kit_r2.tgz" "$EX/s2bres/s2b_min.tgz" \
    "$L/cb2_l1b_code.bundle" "$L/cb2_l1b_code.rev" "$L/cb2_l1b_SHA256SUMS" "$L/cb2_l1b_pod.sh" root@$IP:/workspace/
cp "$L/cb2_sync_l1b_bench.sh" ~/cb2_sync_l1b_bench.sh && chmod +x ~/cb2_sync_l1b_bench.sh
setsid nohup ~/cb2_sync_l1b_bench.sh "$POD" "$IP" "$PORT" > ~/cb2_sync_l1b_bench.out 2>&1 < /dev/null &
echo "WATCHER STARTED: status in $L/bench_status.txt and $L/bench_pod.log every 2 min; it deletes the pod at the end"
timeout 60 $SSH -n "cd /workspace && (PHASE=bench setsid nohup bash /workspace/cb2_l1b_pod.sh > /workspace/l1b-bench.log 2>&1 < /dev/null &) ; sleep 1" || true
sleep 20
timeout 30 $SSH -n 'tail -5 /workspace/l1b-bench.log' || true
echo "BENCH STARTED on $POD ($NAME)"
