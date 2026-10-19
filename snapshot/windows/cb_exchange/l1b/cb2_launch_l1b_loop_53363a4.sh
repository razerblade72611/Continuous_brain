#!/usr/bin/env bash
# WSL: CB2-L1b PHASE=run -- the locked 10-cycle loop on one RTX 4090 (~24 h measured).  Same controller as the bench:
# claim, budget clock from before creation, every call bounded, verified cleanup; on success the loop archive goes to
# ~/cb_data/l1b-bench/cb2_l1b_loop.tgz (WSL-native) and the pod is deleted; on failure or at the cap the pod is STOPPED
# with its volume kept.  Caps (plan C-021): MAXH (default 30 h) and MAX_COST (default $27) at the live rate.
#   bash /mnt/c/Users/davei/ContinuousBrain/cb_exchange/l1b/cb2_launch_l1b_loop.sh                  # community cloud
#   CLOUD=SECURE bash .../cb2_launch_l1b_loop.sh                                                     # secure cloud
# The final battery is a separate phase and a separate decision.
set -euo pipefail
L=/mnt/c/Users/davei/ContinuousBrain/cb_exchange/l1b
exec python3 "$L/cb2_l1b_bench_ctl.py" launch --phase run --cloud "${CLOUD:-COMMUNITY}" --maxh "${MAXH:-30}" \
  --max-cost "${MAX_COST:-27}" --download-timeout "${DL_TIMEOUT:-7200}" "$@"
