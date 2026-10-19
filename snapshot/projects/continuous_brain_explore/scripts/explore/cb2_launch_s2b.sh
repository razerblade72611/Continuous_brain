#!/usr/bin/env bash
# WSL: launch the CB2-S2b pod run end to end.
#   bash /mnt/c/Users/davei/ContinuousBrain/cb_exchange/s2b/cb2_launch_s2b.sh            # create a pod, then upload+start
#   bash .../cb2_launch_s2.sh <pod-id> <ip> <ssh-port>                                   # upload+start on an existing pod
# Uploads tile_decoder_v1.pt, cb2_r2_model.tar, cb2_r3h_results.tgz (from cb_exchange) and the s2 code package (from
# cb_exchange/s2), checks them on the pod, starts cb2_s2b_pod.sh detached, then starts the sync loop (which downloads
# the results, verifies sha256 and only then deletes the pod).
set -euo pipefail
EX=/mnt/c/Users/davei/ContinuousBrain/cb_exchange; S2=$EX/s2b
KEY=/home/davei/.runpod/ssh/runpodctl-ssh-key; RP=~/projects/continuous_brain_explore/runpodctl
GPU=${GPU:-"NVIDIA GeForce RTX 4090"}
cd "$S2" && sha256sum -c cb2_s2b_SHA256SUMS
cd "$EX" && sha256sum -c <<SUMS
0c46b2d1ace00c027d45e23ae01b00da8b802d10660122da49f9727a1cfe9999  tile_decoder_v1.pt
50ec80de0988aeebd2a9797483181e0a5d9890f8a820336621b0a4920c2a2cb6  cb2_r2_model.tar
71f215f61b04af7ec66a97615364f30481f42803f3f9812c5e0c5537bf8438ef  cb2_r3h_results.tgz
SUMS

if [ $# -ge 3 ]; then POD=$1; IP=$2; PORT=$3
else
  echo "== creating pod ($GPU)"
  $RP pod create --name cb2-s2b --gpu-id "$GPU" --image "runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404" \
      --container-disk-in-gb 40 --volume-in-gb 40 --ports "22/tcp" | tee /tmp/cb2_s2b_create.txt
  POD=$(grep -oE '"?id"?[": ]+[a-z0-9]{10,16}' /tmp/cb2_s2b_create.txt | head -1 | grep -oE '[a-z0-9]{10,16}$' || true)
  [ -n "$POD" ] || { echo "Could not read the pod id above. Re-run: bash $0 <pod-id> <ip> <ssh-port>"; exit 1; }
  echo "pod: $POD -- waiting for SSH"
  for i in $(seq 1 60); do
    $RP pod get "$POD" > /tmp/cb2_s2b_get.txt 2>&1 || true
    read -r IP PORT < <(python3 - /tmp/cb2_s2b_get.txt <<'PY'
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
  [ -n "${IP:-}" ] && [ -n "${PORT:-}" ] || { cat /tmp/cb2_s2b_get.txt; echo "SSH not found yet. When it is: bash $0 $POD <ip> <ssh-port>"; exit 1; }
fi
echo "pod $POD  ssh root@$IP -p $PORT"
SSH="ssh -i $KEY -p $PORT -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 root@$IP"
for i in $(seq 1 30); do $SSH true 2>/dev/null && break; sleep 10; done
echo "== uploading"
scp -i $KEY -P $PORT -o StrictHostKeyChecking=accept-new "$EX/tile_decoder_v1.pt" "$EX/cb2_r2_model.tar" "$EX/cb2_r3h_results.tgz" \
    "$S2/cb2_s2b_code.bundle" "$S2/cb2_s2b_code.rev" "$S2/cb2_s2b_SHA256SUMS" "$S2/cb2_s2b_pod.sh" root@$IP:/workspace/
# the sync loop first (so the pod is always watched and deleted at the end), then the run, detached
cp "$S2/cb2_sync_s2b.sh" ~/cb2_sync_s2b.sh && chmod +x ~/cb2_sync_s2b.sh
setsid nohup ~/cb2_sync_s2b.sh "$POD" "$IP" "$PORT" > ~/cb2_sync_s2b.out 2>&1 < /dev/null &
echo "SYNC STARTED: pod $POD ($IP:$PORT); status in $EX/s2b_status.txt and $EX/s2b_pod.log every 5 min"
timeout 60 $SSH -n 'cd /workspace && (setsid nohup bash /workspace/cb2_s2b_pod.sh > /workspace/s2b.log 2>&1 < /dev/null &) ; sleep 1' || true
sleep 20
timeout 30 $SSH -n 'tail -5 /workspace/s2b.log' || true
echo "RUN STARTED on $POD"
