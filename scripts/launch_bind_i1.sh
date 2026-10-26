#!/usr/bin/env bash
# BIND-I1 on a pod: L kl1.0 lr1e-4, dev seeds 2-5 (one worker each, in parallel), S/X/U branched from life 1.
#   bash scripts/launch_bind_i1.sh            (runs in the background; log: bind_i1.log)
cd "$(dirname "$0")/.." || exit 1
findmnt -no SOURCE /workspace | grep -q "networkvolumes/" || { echo "STOP: /workspace is not a network volume"; exit 1; }
if pgrep -f "[s]cripts/bind_i1.py" >/dev/null; then echo "STOP: BIND-I1 already running"; exit 1; fi
if [ -e bind_i1.log ]; then echo "STOP: bind_i1.log exists - not relaunching blindly"; tail -5 bind_i1.log; exit 1; fi
if [ -n "$(git status --porcelain --untracked-files=no -- being scripts/run_v3_dev.py scripts/bind_i1.py scripts/bind_i1_analyze.py)" ]; then
  echo "STOP: uncommitted changes to experiment source"; exit 1; fi
echo "commit: $(git log --oneline -1)"
for s in 2 3 4 5; do                                             # tokenizer-only scorer preflight (Sol); no GPU work
  .venv/bin/python scripts/bind_i1.py --seed $s --preflight || { echo "STOP: tokenizer-prefix preflight failed (seed $s)"; exit 1; }
done
mkdir -p results/dev3/bind_i1
nohup bash -c '
  set -o pipefail
  pids=""
  for s in 2 3 4 5; do
    .venv/bin/python scripts/bind_i1.py --seed $s --out results/dev3/bind_i1/seed$s.json \
        > results/dev3/bind_i1_seed$s.log 2>&1 & pids="$pids $!"
  done
  ex=""; for p in $pids; do wait $p; ex="$ex $?"; done
  tail -n 2 results/dev3/bind_i1_seed*.log
  .venv/bin/python scripts/bind_i1_analyze.py results/dev3/bind_i1/seed*.json > results/dev3/bind_i1_analysis.txt 2>&1
  cat results/dev3/bind_i1_analysis.txt
  echo "{\"exits\": \"$ex\", \"commit\": \"$(git rev-parse HEAD)\"}" > results/dev3/bind_i1_summary.json
  tar czf results_bind_i1.tar.gz results/dev3/bind_i1 results/dev3/bind_i1_*.log results/dev3/bind_i1_summary.json \
      results/dev3/bind_i1_analysis.json results/dev3/bind_i1_analysis.txt
  sha256sum results_bind_i1.tar.gz > results_bind_i1.tar.gz.sha256 && sync
  echo "BUNDLE $(cat results_bind_i1.tar.gz.sha256)"
  echo "BIND-I1 ALL DONE exits=$ex"
' > bind_i1.log 2>&1 &
echo "BIND-I1 LAUNCHED - waiting 150 s"
sleep 150
tail -n 3 results/dev3/bind_i1_seed*.log 2>/dev/null
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
