#!/usr/bin/env bash
# WSL: CB2-L1b PHASE=bench on one RTX 4090 (~1 h).  All logic (claim, verified cleanup, timeouts, billing guard,
# evidence download + verification) lives in cb2_l1b_bench_ctl.py; this only runs it from the exchange folder.
#   bash /mnt/c/Users/davei/ContinuousBrain/cb_exchange/l1b/cb2_launch_l1b_bench.sh                  # community cloud
#   CLOUD=SECURE bash .../cb2_launch_l1b_bench.sh                                                    # secure cloud
# Status: cb_exchange/l1b/bench_status.txt (+ bench_pod.log).  The loop (PHASE=run) is never started by this.
set -euo pipefail
L=/mnt/c/Users/davei/ContinuousBrain/cb_exchange/l1b
exec python3 "$L/cb2_l1b_bench_ctl.py" launch --cloud "${CLOUD:-COMMUNITY}" --maxh "${MAXH:-3}" "$@"
