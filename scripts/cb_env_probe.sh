#!/usr/bin/env bash
# Run on David's PC (WSL). Reads (never changes) the CA-MTL-1 volume's Python environment and stream-tokenizer files
# from a CPU pod, so a volume-free pod elsewhere can reproduce them. Prints everything to ~/cb_env_probe.txt; pod removed.
#   bash cb_env_probe.sh
VOL=29qhoqghqf; DC=CA-MTL-1; K=$HOME/.runpod/ssh/runpodctl-ssh-key; OUT=$HOME/cb_env_probe.txt
runpodctl pod create --name cb-env-probe --compute-type CPU --template-id runpod-torch-v280 --data-center-ids $DC \
  --network-volume-id $VOL --volume-mount-path /workspace --ssh --wait --wait-timeout 15m > ~/cb_probe_pod.json \
  || { echo "STOP: CPU pod create failed"; cat ~/cb_probe_pod.json; exit 1; }
POD=$(python3 -c 'import json;print(json.load(open("'"$HOME"'/cb_probe_pod.json"))["id"])') || exit 1
trap 'echo "== removing pod $POD"; runpodctl pod remove $POD >/dev/null' EXIT
eval "$(runpodctl ssh info $POD | python3 -c 'import sys,json;d=json.load(sys.stdin);print("IP=%s PORT=%s" % (d["ip"], d["port"]))')"
ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -i $K -p $PORT root@$IP 'bash -s' > $OUT 2>&1 <<'REMOTE'
cd /workspace/Continuous_brain || exit 1
echo "== image python: $(python3 --version 2>&1) at $(readlink -f $(which python3))"
echo "== pyvenv.cfg"; cat .venv/pyvenv.cfg
echo "== venv python: $(.venv/bin/python --version 2>&1) -> $(readlink -f .venv/bin/python)"
echo "== freeze"; .venv/bin/python -m pip freeze --all 2>&1 || ls .venv/lib/python3*/site-packages | grep dist-info
echo "== torch"; .venv/bin/python -c "import torch,transformers;print(torch.__version__, torch.version.cuda, transformers.__version__)" 2>&1
echo "== stream tokenizer files (models/qwen3-0.6b)"; (cd models/qwen3-0.6b && ls -la && sha256sum $(find . -maxdepth 1 -type f | sort))
echo "== hf metadata"; find models/qwen3-0.6b /workspace/hf/Qwen3-8B -path '*.cache*' -type f 2>/dev/null | head -20 | while read f; do echo "$f: $(head -c 200 "$f")"; done
echo "== models dir"; ls -la models /workspace/hf 2>&1
echo "== PROBE DONE"
REMOTE
cat $OUT; grep -q 'PROBE DONE' $OUT && echo "== saved to $OUT" || { echo "== probe incomplete"; exit 1; }
