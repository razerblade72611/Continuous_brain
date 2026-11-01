#!/usr/bin/env bash
# Regression for the ABST-0 launchers' detached-boot polling (Sol, 2026-11-01): the deadline holds on EVERY iteration
# (persistent unreachable polls and reachable-but-silent boots both time out), upload and start are verified, STOP and
# BOOT OK are honoured.  The real launcher block runs with fake ssh (local bash), a 1-s poll and a 4-s deadline.
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd); fail() { echo "FAIL: $*"; exit 1; }
for L in scripts/cb_abst0_p1.sh scripts/cb_abst0_p2.sh; do
  T=$(mktemp -d); mkdir -p $T/ws $T/out
  awk '/^# boot runs detached on the pod/{f=1} f{print} /^done$/ && f{exit}' $REPO/$L | sed "s|/workspace|$T/ws|g" > $T/poll.sh
  [ -s $T/poll.sh ] || fail "$L: boot block not found"
  ok_ssh() { printf '#!/usr/bin/env bash\nbash -c "$*"\n' > $T/ssh; chmod +x $T/ssh; }
  run() {  # $1 = boot.sh body (empty = no boot.sh)
    rm -f $T/ws/boot.log $T/ws/boot.sh; [ -n "$1" ] && printf '%s\n' "$1" > $T/ws/boot.sh
    S=$T/ssh OUT=$T/out D=x PIN=y REV=z BOOT_POLL=1 BOOT_DEADLINE=4 timeout 20 bash -c 'source "$0"; echo "RESULT reached-end"' $T/poll.sh 2>&1 | tail -3
    [ "${PIPESTATUS[0]}" = 124 ] && echo "RESULT hung (no deadline)"; }
  ok_ssh
  r=$(run "exec > $T/ws/boot.log 2>&1; echo 'ENV x'; sleep 1; echo 'BOOT OK'"); [[ $r == *"RESULT reached-end"* ]] || fail "$L: success: $r"
  r=$(run "exec > $T/ws/boot.log 2>&1; echo 'STOP: boot failed (exit 1)'"); [[ $r == *"STOP: boot failed - see"* && $r != *reached-end* ]] || fail "$L: STOP: $r"
  r=$(run "exec > $T/ws/boot.log 2>&1; sleep 30"); [[ $r == *"did not finish"* && $r != *reached-end* ]] || fail "$L: reachable but silent must time out: $r"
  r=$(run ""); [[ $r == *"did not upload intact"* ]] || fail "$L: missing boot.sh must stop: $r"
  r=$(run "exec > $T/ws/boot.log 2>&1; if then"); [[ $r == *"did not upload intact"* ]] || fail "$L: corrupt boot.sh must stop: $r"
  # start fails: boot.sh never creates boot.log
  r=$(run "sleep 1"); [[ $r == *"did not start"* ]] || fail "$L: start without boot.log must stop: $r"
  # persistent unreachable after a successful start: every poll fails -> deadline
  printf '#!/usr/bin/env bash\nif [[ "$*" == *__POLL_OK__* ]]; then exit 255; fi\nbash -c "$*"\n' > $T/ssh
  r=$(run "exec > $T/ws/boot.log 2>&1; sleep 30"); [[ $r == *"did not finish"* && $r != *reached-end* ]] || fail "$L: persistent unreachable must time out: $r"
  # transient unreachable then success
  printf '#!/usr/bin/env bash\nn=$(cat %s/n 2>/dev/null || echo 0); n=$((n+1)); echo $n > %s/n\nif [[ "$*" == *__POLL_OK__* ]] && [ $n = 3 ]; then exit 255; fi\nbash -c "$*"\n' $T $T > $T/ssh
  rm -f $T/n; r=$(run "exec > $T/ws/boot.log 2>&1; sleep 2; echo 'BOOT OK'"); [[ $r == *"reached-end"* ]] || fail "$L: transient unreachable then success: $r"
  rm -rf "$T"
done
echo "PASS: ABST-0 boot polling (success, STOP, reachable-silent timeout, missing/corrupt boot.sh, no start, persistent-unreachable timeout, transient unreachable) x 2 launchers"
