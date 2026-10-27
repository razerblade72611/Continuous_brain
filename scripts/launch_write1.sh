#!/usr/bin/env bash
# WRITE-1 on a pod: dev seeds 2-5 (one worker each, in parallel), arms A/R/P branched from the BIND-I1 life-1 state.
#   bash scripts/launch_write1.sh            (runs in the background; log: write1.log)
cd "$(dirname "$0")/.." || exit 1
findmnt -no SOURCE /workspace | grep -q "networkvolumes/" || { echo "STOP: /workspace is not a network volume"; exit 1; }
if pgrep -f "[s]cripts/write1.py" >/dev/null; then echo "STOP: WRITE-1 already running"; exit 1; fi
if [ -e write1.log ]; then echo "STOP: write1.log exists - not relaunching blindly"; tail -5 write1.log; exit 1; fi
if [ -n "$(git status --porcelain --untracked-files=no -- being scripts/run_v3_dev.py scripts/bind_i1.py scripts/decay1.py scripts/write1.py scripts/write1_analyze.py results/dev3/bind_i1 results/dev3/decay1)" ]; then
  echo "STOP: uncommitted changes to experiment source or the BIND-I1 reference"; exit 1; fi
for s in 2 3 4 5; do [ -f results/dev3/bind_i1/seed$s.json ] && [ -f results/dev3/decay1/seed$s.json ] || { echo "STOP: reference seed$s missing"; exit 1; }; done
echo "commit: $(git log --oneline -1)"
for s in 2 3 4 5; do
  .venv/bin/python scripts/write1.py --seed $s --preflight || { echo "STOP: tokenizer-prefix preflight failed (seed $s)"; exit 1; }
done
mkdir -p results/dev3/write1
nohup bash -c '
  set -o pipefail
  pids=""
  for s in 2 3 4 5; do
    .venv/bin/python scripts/write1.py --seed $s --out results/dev3/write1/seed$s.json \
        --reference results/dev3/bind_i1/seed$s.json --decay-reference results/dev3/decay1/seed$s.json \
        > results/dev3/write1_seed$s.log 2>&1 & pids="$pids $!"
  done
  ex=""; for p in $pids; do wait $p; ex="$ex $?"; done
  tail -n 2 results/dev3/write1_seed*.log
  .venv/bin/python scripts/write1_analyze.py results/dev3/write1/seed[0-9].json > results/dev3/write1_analysis.txt 2>&1
  cat results/dev3/write1_analysis.txt
  echo "{\"exits\": \"$ex\", \"commit\": \"$(git rev-parse HEAD)\"}" > results/dev3/write1_summary.json
  sha256sum results/dev3/write1/*_states.pt > results/dev3/write1_states.sha256 2>/dev/null
  tar czf results_write1.tar.gz --exclude="*_states.pt" results/dev3/write1 results/dev3/write1_*.log \
      results/dev3/write1_summary.json results/dev3/write1_analysis.json results/dev3/write1_analysis.txt \
      results/dev3/write1_states.sha256
  sha256sum results_write1.tar.gz > results_write1.tar.gz.sha256 && sync
  echo "BUNDLE $(cat results_write1.tar.gz.sha256)"
  echo "WRITE-1 ALL DONE exits=$ex"
' > write1.log 2>&1 < /dev/null &
echo "WRITE-1 LAUNCHED - waiting 120 s"
sleep 120
tail -n 2 results/dev3/write1_seed*.log 2>/dev/null
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
