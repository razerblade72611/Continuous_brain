#!/usr/bin/env bash
# Regression for the ABST-0 launchers' job upload / start / poll (2026-11-01: an unchecked ssh call left phase 1's job
# unstarted while the poll loop printed "No such file" forever). Real launcher blocks, fake ssh = local bash.
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd); fail() { echo "FAIL: $*"; exit 1; }
for ph in 1 2; do
  L=$REPO/scripts/cb_abst0_p$ph.sh; T=$(mktemp -d); mkdir -p $T/ws
  # job block: from the local job-file heredoc through the start confirmation; then the poll loop
  awk '/^JOBF=\$\(mktemp\); cat > \$JOBF <<.JOB.$/{f=1} f{print} /^case "\$st" in STARTED\|EXISTS\) echo "== job started"/ && f{exit}' $L \
    | sed "s|/workspace|$T/ws|g" > $T/job.sh
  awk '/^t0=\$\(date \+%s\)$/{f=1} f{print} /^done$/ && f{exit}' $L | sed "s|/workspace|$T/ws|g" > $T/poll.sh
  grep -q 'echo "== job started"' $T/job.sh && grep -q 'NOLOG' $T/poll.sh || fail "p$ph: blocks not found"
  # the job itself: write a fake job that logs and finishes
  sed -i "s|^cd /workspace/Continuous_brain_abst0$|cd $T|" $T/job.sh
  ok() { printf '#!/usr/bin/env bash\nbash -c "$*"\n' > $T/ssh; chmod +x $T/ssh; }
  runjob() { rm -f $T/ws/abst0_p$ph.log $T/ws/abst0_p$ph.sh; S=$T/ssh RETRY_WAIT=0 timeout 30 bash $T/job.sh 2>&1 | tail -2; }
  runpoll() { S=$T/ssh JOB_POLL=1 JOB_CAP=5 timeout 30 bash $T/poll.sh 2>&1 | tail -2; [ "${PIPESTATUS[0]}" = 124 ] && echo "HUNG"; }
  ok; r=$(runjob); [[ $r == *"== job started"* ]] || fail "p$ph: normal start: $r"
  sleep 1; [ -f $T/ws/abst0_p$ph.log ] || fail "p$ph: job log missing after start"
  # job already started (log exists) -> never started twice
  echo running > $T/ws/abst0_p$ph.log
  r=$(S=$T/ssh RETRY_WAIT=0 timeout 30 bash -c "$(sed -n '/^st=""$/,/^case "\$st"/p' $T/job.sh)" 2>&1); [[ $r == *"job started"* ]] && [ "$(cat $T/ws/abst0_p$ph.log)" = running ] || fail "p$ph: existing log must not restart: $r"
  # upload never verifies -> STOP
  printf '#!/usr/bin/env bash\nif [[ "$*" == "cat > "* ]]; then cat > /dev/null; exit 0; fi\nbash -c "$*"\n' > $T/ssh
  r=$(runjob); [[ $r == *"could not be uploaded intact"* && $r != *"job started"* ]] || fail "p$ph: unverified upload must stop: $r"
  # upload fails once, then succeeds
  printf '#!/usr/bin/env bash\nn=$(cat %s/n 2>/dev/null || echo 0); n=$((n+1)); echo $n > %s/n\nif [ $n = 1 ]; then exit 255; fi\nbash -c "$*"\n' $T $T > $T/ssh
  rm -f $T/n; r=$(runjob); [[ $r == *"== job started"* ]] || fail "p$ph: transient upload failure then success: $r"
  # start never confirmed (ssh fails for the start command) -> STOP
  printf '#!/usr/bin/env bash\nif [[ "$*" == *nohup* ]]; then exit 255; fi\nbash -c "$*"\n' > $T/ssh
  r=$(runjob); [[ $r == *"job did not start"* ]] || fail "p$ph: unconfirmed start must stop: $r"
  # poll: DONE -> proceeds; reachable without log 3x -> STOP; persistently unreachable -> cap
  ok; echo "ABST0 P$ph DONE exit=0" > $T/ws/abst0_p$ph.log; r=$(runpoll); [[ $r == *"== done"* ]] || fail "p$ph: DONE: $r"
  rm -f $T/ws/abst0_p$ph.log; r=$(runpoll); [[ $r == *"job log is missing"* ]] || fail "p$ph: missing log must stop: $r"
  printf '#!/usr/bin/env bash\nexit 255\n' > $T/ssh; r=$(runpoll); [[ $r == *"cap reached"* && $r != *HUNG* ]] || fail "p$ph: unreachable must hit the cap: $r"
  rm -rf "$T"
done
echo "PASS: ABST-0 job upload/start/poll (start, no double start, unverified upload, transient upload failure, unconfirmed start, DONE, missing log, unreachable cap) x 2 launchers"
