#!/usr/bin/env bash
# BIND-D1 on a pod: L kl1.0 lr1e-4 and S layer21 (A0 configs), seeds 0,1, 5 lives, two workers in parallel.
#   bash scripts/launch_bind_d1.sh            (runs in the background; log: bind_d1.log)
cd "$(dirname "$0")/.." || exit 1
findmnt -no SOURCE /workspace | grep -q "networkvolumes/" || { echo "STOP: /workspace is not a network volume"; exit 1; }
if pgrep -f "scripts/bind_d1.py" >/dev/null; then echo "STOP: BIND-D1 already running"; exit 1; fi
if [ -e bind_d1.log ]; then echo "STOP: bind_d1.log exists - not relaunching blindly"; tail -5 bind_d1.log; exit 1; fi
if [ -n "$(git status --porcelain --untracked-files=no -- being scripts/run_v3_dev.py scripts/bind_d1.py)" ]; then
  echo "STOP: uncommitted changes to experiment source"; exit 1; fi
echo "commit: $(git log --oneline -1)"
nohup bash -c '
  set -o pipefail
  A0=results/dev3/stageA2_A0
  .venv/bin/python scripts/bind_d1.py --arm L --set lr=1e-4 kl_weight=1.0 night_passes=4 --seeds 0,1 --lives 5 \
      --out results/dev3/bind_d1/L_kl1.0_lr1e-4.json --reference $A0/L_kl_weight1.0_lr0.0001_night_passes4.json \
      > results/dev3/bind_d1_L.log 2>&1 & p1=$!
  .venv/bin/python scripts/bind_d1.py --arm S --set layer=21 lr=100 top_t=64 s_fit=true night_passes=4 --seeds 0,1 --lives 5 \
      --out results/dev3/bind_d1/S_layer21.json --reference $A0/S_layer21_lr100_night_passes4_s_fittrue_top_t64.json \
      > results/dev3/bind_d1_S.log 2>&1 & p2=$!
  wait $p1; e1=$?; wait $p2; e2=$?
  tail -2 results/dev3/bind_d1_L.log results/dev3/bind_d1_S.log
  echo "{\"L_exit\": $e1, \"S_exit\": $e2, \"commit\": \"$(git rev-parse HEAD)\"}" > results/dev3/bind_d1_summary.json
  tar czf results_bind_d1.tar.gz results/dev3/bind_d1 results/dev3/bind_d1_*.log results/dev3/bind_d1_summary.json
  sha256sum results_bind_d1.tar.gz > results_bind_d1.tar.gz.sha256 && sync
  echo "BUNDLE $(cat results_bind_d1.tar.gz.sha256)"
  echo "BIND-D1 ALL DONE L_exit=$e1 S_exit=$e2"
' > bind_d1.log 2>&1 &
echo "BIND-D1 LAUNCHED - waiting 120 s"
sleep 120
tail -3 results/dev3/bind_d1_L.log results/dev3/bind_d1_S.log 2>/dev/null
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
