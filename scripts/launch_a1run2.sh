#!/usr/bin/env bash
# Launch A2 Phase 1 run 2 (A0 INCONCLUSIVE action): 10 A1 configs, seeds 0,1, 25 lives, 5 workers, self-stop.
#   bash scripts/launch_a1run2.sh
cd "$(dirname "$0")/.." || exit 1
findmnt -no SOURCE /workspace | grep -q "networkvolumes/" || { echo "STOP: /workspace is not a network volume"; exit 1; }
if pgrep -f "scripts/stage_v3.py" >/dev/null; then echo "STOP: a stage is already running"; pgrep -af stage_v3; exit 1; fi
if [ -e stageA1run2.log ]; then echo "STOP: stageA1run2.log already exists - not relaunching blindly"; tail -5 stageA1run2.log; exit 1; fi
echo "commit: $(git log --oneline -1)"
nohup .venv/bin/python scripts/stage_v3.py A --only A1 --parallel 5 --stop-pod > stageA1run2.log 2>&1 &
echo "PHASE 1 LAUNCHED (pid $!) - waiting 150 s for the first workers"
sleep 150
tail -8 stageA1run2.log
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
