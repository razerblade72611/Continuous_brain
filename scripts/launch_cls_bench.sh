#!/usr/bin/env bash
# CLS-0 resource benchmark on a pod: download Qwen3-8B + Qwen3-Embedding-0.6B to the volume (once), then benchmark the
# episodic memory on Qwen3-8B (bf16, published rank-2 setting, with the external embedding model) and on our
# Qwen3-0.6B (fp32, continuity). Runs in the background; log: cls_bench.log
cd "$(dirname "$0")/.." || exit 1
findmnt -no SOURCE /workspace | grep -q "networkvolumes/" || { echo "STOP: /workspace is not a network volume"; exit 1; }
if pgrep -f "[s]cripts/cls_bench.py" >/dev/null; then echo "STOP: CLS bench already running"; exit 1; fi
if [ -e cls_bench.log ]; then echo "STOP: cls_bench.log exists - not relaunching blindly"; tail -5 cls_bench.log; exit 1; fi
if [ -n "$(git status --porcelain --untracked-files=no -- being scripts/cls_bench.py)" ]; then
  echo "STOP: uncommitted changes to experiment source"; exit 1; fi
HF=/workspace/hf
need=0; [ -f $HF/Qwen3-8B/config.json ] || need=$((need + 20)); [ -f $HF/Qwen3-Embedding-0.6B/config.json ] || need=$((need + 2))
free=$(df -BG --output=avail /workspace | tail -1 | tr -dc 0-9)
echo "volume free: ${free} GB, download needs ~${need} GB"
[ "$free" -gt $((need + 5)) ] || { echo "STOP: not enough free space on the volume"; exit 1; }
echo "commit: $(git log --oneline -1)"
mkdir -p results/cls
nohup bash -c '
  set -o pipefail
  HF=/workspace/hf
  .venv/bin/python -c "
from huggingface_hub import snapshot_download
for r in (\"Qwen/Qwen3-8B\", \"Qwen/Qwen3-Embedding-0.6B\"):
    p = snapshot_download(r, local_dir=\"$HF/\" + r.split(\"/\")[1], allow_patterns=[\"*.json\", \"*.safetensors\", \"*.txt\", \"*.model\", \"tokenizer*\", \"merges.txt\", \"vocab.json\"])
    print(\"downloaded\", p, flush=True)
" || { echo "DOWNLOAD FAILED"; echo "CLS BENCH ALL DONE exits=download_failed"; exit 1; }
  .venv/bin/python scripts/cls_bench.py --model $HF/Qwen3-8B --dtype bf16 --external $HF/Qwen3-Embedding-0.6B \
      --out results/cls/bench_qwen3_8b.json > results/cls/bench_qwen3_8b.log 2>&1; e1=$?
  .venv/bin/python scripts/cls_bench.py --model models/qwen3-0.6b --dtype fp32 \
      --out results/cls/bench_qwen3_0.6b.json > results/cls/bench_qwen3_0.6b.log 2>&1; e2=$?
  tail -n 3 results/cls/bench_*.log
  tar czf results_cls_bench.tar.gz results/cls
  sha256sum results_cls_bench.tar.gz > results_cls_bench.tar.gz.sha256 && sync
  echo "BUNDLE $(cat results_cls_bench.tar.gz.sha256)"
  echo "CLS BENCH ALL DONE exits=$e1 $e2"
' > cls_bench.log 2>&1 < /dev/null &
echo "CLS BENCH LAUNCHED - waiting 60 s"
sleep 60
tail -n 3 cls_bench.log
