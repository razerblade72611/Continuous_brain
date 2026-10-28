#!/usr/bin/env bash
# CLS-0B on a pod: one worker per seed, in parallel (default pilot seeds "2 3"), then the frozen analysis.
#   bash scripts/launch_cls0b.sh "2 3"          (runs in the background; log: cls0b.log)
# A worker that is restarted resumes from its saved episode bank (digest-checked).
SEEDS_ARG=${1:-2 3}
cd "$(dirname "$0")/.." || exit 1
findmnt -no SOURCE /workspace | grep -q "networkvolumes/" || { echo "STOP: /workspace is not a network volume"; exit 1; }
if pgrep -f "[s]cripts/cls0b.py" >/dev/null; then echo "STOP: CLS-0B already running"; exit 1; fi
if [ -e cls0b.log ]; then echo "STOP: cls0b.log exists - not relaunching blindly"; tail -5 cls0b.log; exit 1; fi
if [ -n "$(git status --porcelain --untracked-files=no -- being scripts/bind_i1.py scripts/cls0b.py scripts/cls0b_analyze.py results/dev3/bind_i1)" ]; then
  echo "STOP: uncommitted changes to experiment source"; exit 1; fi
HF=/workspace/hf
for m in Qwen3-8B Qwen3-Embedding-0.6B; do [ -f $HF/$m/config.json ] || { echo "STOP: $HF/$m missing"; exit 1; }; done
.venv/bin/python -m pytest -q tests/test_cls0b.py >/dev/null 2>&1 || echo "note: pytest unavailable or failing in .venv (unit tests are run before freezing)"
echo "commit: $(git log --oneline -1)"; echo "seeds: $SEEDS_ARG"
mkdir -p results/cls0b
nohup bash -c '
  set -o pipefail
  HF=/workspace/hf
  pids=""
  for s in '"$SEEDS_ARG"'; do
    .venv/bin/python scripts/cls0b.py --seed $s --model $HF/Qwen3-8B --external $HF/Qwen3-Embedding-0.6B \
        --out results/cls0b/seed$s.json > results/cls0b/seed$s.log 2>&1 & pids="$pids $!"
    sleep 90                                     # stagger model loading
  done
  ex=""; bad=0; for p in $pids; do wait $p; e=$?; ex="$ex $e"; [ $e -eq 0 ] || bad=1; done
  tail -n 2 results/cls0b/seed*.log
  .venv/bin/python scripts/cls0b_analyze.py results/cls0b/seed[0-9].json > results/cls0b/analysis.txt 2>&1
  cat results/cls0b/analysis.txt
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
  tar czf results_cls0b.tar.gz results/cls0b
  sha256sum results_cls0b.tar.gz > results_cls0b.tar.gz.sha256 && sync
  echo "BUNDLE $(cat results_cls0b.tar.gz.sha256)"
  if [ $bad -eq 0 ]; then st=CLS0B_COMPLETE; else st=CLS0B_FAILED; fi
  echo "CLS-0B ALL DONE status=$st exits=$ex"
' > cls0b.log 2>&1 < /dev/null &
echo "CLS-0B LAUNCHED - waiting 240 s"
sleep 240
tail -n 2 results/cls0b/seed*.log 2>/dev/null
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
