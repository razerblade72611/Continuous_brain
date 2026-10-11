#!/usr/bin/env bash
# WSL: stop the running (NaN-contaminated) L1 on the existing cb2-l1 pod, print the diagnosis, upload the fixed code
# (cb_exchange/l1, verified), restart L1 fresh on the same pod, and restart the download loop.
#   bash /mnt/c/Users/davei/ContinuousBrain/cb_exchange/l1/cb2_l1_restart.sh
set -uo pipefail
cd ~/projects/continuous_brain_explore || exit 1
RP=./runpodctl; K=/home/davei/.runpod/ssh/runpodctl-ssh-key; X=/mnt/c/Users/davei/ContinuousBrain/cb_exchange
(cd $X/l1 && sha256sum -c cb2_l1_SHA256SUMS) || { echo "STOP: package checksum"; exit 1; }
$RP pod list > /tmp/cb2_pods.json
read -r POD N < <(python3 - /tmp/cb2_pods.json <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); d = d if isinstance(d, list) else d.get("pods", [])
p = [x for x in d if x.get("name") == "cb2-l1"]
print(p[0]["id"] if p else "", len(p))
PY
)
[ -n "$POD" ] || { echo "no cb2-l1 pod found"; cat /tmp/cb2_pods.json; exit 1; }
[ "$N" = "1" ] || { echo "STOP: $N cb2-l1 pods -- delete the extra one(s) first"; cat /tmp/cb2_pods.json; exit 1; }
$RP pod get "$POD" > /tmp/cb2_get.json
read -r IP PORT < <(python3 - /tmp/cb2_get.json <<'PY'
import json, re, sys
t = open(sys.argv[1]).read(); ip = port = None
try:
    d = json.loads(t); d = d[0] if isinstance(d, list) else d
    ip = d.get("publicIp") or (d.get("runtime") or {}).get("publicIp")
    pm = d.get("portMappings") or {}; port = pm.get("22") or pm.get(22)
    for p in ((d.get("runtime") or {}).get("ports") or []):
        if p.get("privatePort") == 22 and p.get("isIpPublic"): ip, port = p.get("ip"), p.get("publicPort")
except Exception:
    pass
if not (ip and port):
    m = re.search(r"root@(\d+\.\d+\.\d+\.\d+)\s+-p\s+(\d+)", t) or re.search(r"(\d+\.\d+\.\d+\.\d+):(\d+)\s*->\s*22", t)
    if m: ip, port = m.group(1), m.group(2)
print(ip or "", port or "")
PY
)
[ -n "$IP" ] && [ -n "$PORT" ] || { echo "could not read ssh address"; cat /tmp/cb2_get.json; exit 1; }
echo "== pod $POD  $IP:$PORT"
SSH="ssh -i $K -p $PORT -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 root@$IP"
pkill -f cb2_sync_l1.sh
echo "== stopping the run + diagnosis"
$SSH 'bash -s' <<'EOF'
pkill -f cb2_l1_pod.sh; pkill -f "cb2_l1.py run"; sleep 3
cd /workspace/Continuous_brain && /workspace/venv/bin/python - <<'PY'
import io, json, glob, torch, numpy as np, pyarrow.parquet as pq
from pathlib import Path
for t in ("learning", "reference"):
    h = Path(f"runs/explore/cb2/l1/twins/{t}/cb2-ind-0001"); m = json.loads((h / "individual.json").read_text())
    s = torch.load(io.BytesIO((h / "weights" / (m["weights_sha256"] + ".pt")).read_bytes()), map_location="cpu", weights_only=False)
    bad = []
    def walk(x, p):
        if torch.is_tensor(x):
            if x.is_floating_point() and not torch.isfinite(x).all(): bad.append(p)
        elif isinstance(x, dict):
            for k, v in x.items(): walk(v, p + "." + str(k))
    walk(s, ""); print(t, "non-finite tensors:", len(bad), bad[:8])
n = 0
for f in glob.glob("runs/explore/cb2/l1/archive-learning/*/shard-*.student.parquet"):
    r = np.asarray(pq.read_table(f, columns=["reward"])["reward"].to_pylist(), float); n += int((~np.isfinite(r)).sum())
print("non-finite rewards in its lives:", n)
print("cycles recorded:", sum(1 for _ in open("runs/explore/cb2/l1/cycles.jsonl")))
PY
EOF
echo "== uploading the fix"
scp -q -i $K -P $PORT $X/l1/cb2_l1_code.bundle $X/l1/cb2_l1_code.rev $X/l1/cb2_l1_SHA256SUMS $X/l1/cb2_l1_pod.sh root@$IP:/workspace/ || exit 1
$SSH 'cd /workspace && (setsid nohup bash /workspace/cb2_l1_pod.sh >> /workspace/l1.log 2>&1 < /dev/null &); sleep 90; tail -7 /workspace/l1.log'
cp $X/l1/cb2_sync_l1.sh ~/cb2_sync_l1.sh && chmod +x ~/cb2_sync_l1.sh
setsid nohup ~/cb2_sync_l1.sh "$POD" "$IP" "$PORT" > ~/cb2_sync_l1.out 2>&1 < /dev/null &
echo "== RESTARTED on $POD; download loop running"
