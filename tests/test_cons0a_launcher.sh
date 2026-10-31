#!/usr/bin/env bash
# Regression test for scripts/cb_cons0a_run.sh: the REAL remote preflight + job script run against a local clone of this repo
# in a sandbox ("/workspace" -> temp dir); only the GPU stages (scripts/cons0a.py subcommands), findmnt and pgrep are stubbed.
# Also the real PC-side tail with a fake ssh/runpodctl.        bash tests/test_cons0a_launcher.sh
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd); L=$REPO/scripts/cb_cons0a_run.sh
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
PIN=$(grep -o '^PIN=[0-9a-f]*' "$L" | cut -d= -f2)
[ ${#PIN} = 40 ] || { echo "FAIL: launcher PIN is not a full commit hash"; exit 1; }
mkdir -p $T/bin
printf '#!/bin/sh\necho "srv:/networkvolumes/$VOL"\n' > $T/bin/findmnt; printf '#!/bin/sh\nexit 1\n' > $T/bin/pgrep; chmod +x $T/bin/*
awk '/bash -s" <<'"'"'REMOTE'"'"'/{f=1;next} /^REMOTE$/{f=0} f' "$L" > $T/remote.raw
[ -s $T/remote.raw ] || { echo "FAIL: could not extract the remote script"; exit 1; }
git clone -q "file://$REPO" $T/proto && git -C $T/proto checkout -q being/v3-bind 2>/dev/null
cat > $T/fakepy <<'PY'
#!/usr/bin/env bash
if [ "${1:-}" = "-" ]; then
  if [ "${FAKE_MANIFEST:-real}" = ok ]; then cat >/dev/null; echo "MODEL MANIFEST OK"; exit 0; fi
  exec python3 -
fi
[ "${1:-}" = "-c" ] && exec python3 "$@"
out=; prev=; seed=; for x in "$@"; do [ "$prev" = --out ] && out=$x; [ "$prev" = --seed ] && seed=$x; prev=$x; done
case "$2" in
  calibrate) echo "{\"chosen_E\": 8, \"contract_revision_needed\": ${CAL_REV:-false}, \"projection_s\": ${CAL_PROJ:-1000}}" > "$out";;
  prep) echo '{"seed": '"$seed"'}' > "$out";;
  qa) echo '{}' > "$out";;
  run) s=$(basename "$out" | tr -dc 0-9); [ "${FAIL_SEED:-}" = "$s" ] && { echo "run failed (stub)"; exit 3; }
       echo '{}' > "$out/run_seed$s.json"
       for f in V_shared_s1 S-V_s5 I-V_s5 QA_shared_s1 S-QA_s5 I-QA_s5; do head -c 100 /dev/urandom > "$out/cortex_$f.pt"; done;;
  fresh) echo '{}' > "$out";;
  analyze) echo '{"status": "CONS0A_NO_ADVANCE"}' > "$out"; echo "status=CONS0A_NO_ADVANCE";;
  *) exec python3 "$@";;
esac
PY
chmod +x $T/fakepy
setup() {
  rm -rf $T/ws; mkdir -p $T/ws/Continuous_brain/.venv/bin $T/ws/hf/Qwen3-8B $T/ws/Continuous_brain/models/qwen3-0.6b
  echo '{}' > $T/ws/Continuous_brain/models/qwen3-0.6b/tokenizer.json
  cp -a $T/proto $T/ws/Continuous_brain_cons0a
  cp $T/fakepy $T/ws/Continuous_brain/.venv/bin/python
  echo '{}' > $T/ws/hf/Qwen3-8B/config.json
  sed "s|/workspace|$T/ws|g" $T/remote.raw > $T/remote.sh
}
remote() {  # preflight (+ launch); waits for the job; prints "preflight_exit job_exit"
  VOL=29qhoqghqf D=$T/ws/Continuous_brain_cons0a PIN=$PIN CB_UNLOCK=${CB_UNLOCK:-0} PATH=$T/bin:$PATH \
    bash $T/remote.sh > $T/pre.txt 2>&1; local pe=$?
  local je=none
  if [ $pe = 0 ]; then
    for _ in $(seq 120); do grep -q 'CONS-0A JOB DONE' $T/ws/cons0a.log 2>/dev/null && break; sleep 0.25; done
    je=$(grep -o 'CONS-0A JOB DONE exit=[0-9]*' $T/ws/cons0a.log | cut -d= -f2)
  fi
  echo "$pe $je"
}
fail() { echo "FAIL: $*"; echo "--- preflight"; cat $T/pre.txt; echo "--- job log"; cat $T/ws/cons0a.log 2>/dev/null; exit 1; }
R=$T/ws/Continuous_brain_cons0a/results/cons0a

# 1 success: calibrate -> 4 seeds (prep, qa, run, 8 fresh) -> analyze; bundle without .pt; lock released
setup; r=$(FAKE_MANIFEST=ok remote); [ "$r" = "0 0" ] || fail "success path gave '$r'"
for s in 2 3 4 5; do [ $(ls $R/seed$s/fresh_seed${s}_*_s*.json | wc -l) = 8 ] || fail "seed $s: fresh evaluations missing"; done
[ -f $R/analysis.json ] || fail "no analysis"
tar tzf $T/ws/results_cons0a.tar.gz > $T/listing.txt || fail "bundle unreadable"   # no pipe: grep -q + pipefail = SIGPIPE flake
grep -q '\.pt$' $T/listing.txt && fail "bundle contains .pt checkpoints"
grep -q 'analysis.json' $T/listing.txt || fail "bundle lacks analysis.json"
(cd $T/ws && sha256sum -c --quiet results_cons0a.tar.gz.sha256) || fail "bundle checksum"
[ ! -e $T/ws/cons0a.lock ] || fail "lock not released"
# 2 relaunch after success: every seed skipped, still completes
r=$(FAKE_MANIFEST=ok remote); [ "$r" = "0 0" ] || fail "relaunch gave '$r'"
[ $(grep -c 'complete - skipped' $T/ws/cons0a.log) = 4 ] || fail "completed seeds not skipped"
# 3 contract revision needed -> exit 4, no seed started
setup; r=$(FAKE_MANIFEST=ok CAL_REV=true remote); [ "$r" = "0 4" ] && [ ! -d $R/seed2 ] || fail "contract STOP gave '$r'"
grep -q 'revise it once' $T/ws/cons0a.log || fail "contract STOP message"
# 4 projection > 6 h -> exit 4
setup; r=$(FAKE_MANIFEST=ok CAL_PROJ=30000 remote); [ "$r" = "0 4" ] && [ ! -d $R/seed2 ] || fail "projection STOP gave '$r'"
# 5 a failing seed -> job exit 1, no analysis; relaunch redoes that seed from scratch and keeps the completed one
setup; r=$(FAKE_MANIFEST=ok FAIL_SEED=3 remote); [ "$r" = "0 1" ] && [ ! -f $R/analysis.json ] || fail "seed failure gave '$r'"
touch $R/seed3/stale_marker
r=$(FAKE_MANIFEST=ok remote); [ "$r" = "0 0" ] || fail "resume after failure gave '$r'"
grep -q '== seed 2 complete - skipped' $T/ws/cons0a.log || fail "completed seed 2 not skipped on resume"
[ ! -e $R/seed3/stale_marker ] || fail "incomplete seed 3 not restarted from scratch"
# 6 model files differ (REAL manifest code on a fake model dir) -> STOP before launch
setup; r=$(remote); [ "${r%% *}" != 0 ] && grep -q 'MISMATCH' $T/pre.txt && [ ! -e $T/ws/cons0a.log ] || fail "model mismatch ('$r')"
# 7 lock held -> STOP; CB_UNLOCK=1 proceeds
setup; mkdir $T/ws/cons0a.lock; echo otherpod > $T/ws/cons0a.lock/owner
r=$(FAKE_MANIFEST=ok remote); [ "${r%% *}" != 0 ] && grep -q 'lock held by otherpod' $T/pre.txt || fail "held lock ('$r')"
r=$(FAKE_MANIFEST=ok CB_UNLOCK=1 remote); [ "$r" = "0 0" ] || fail "CB_UNLOCK ('$r')"
# 8 corrupted CLS-0B bank -> STOP
setup; echo x >> $T/ws/Continuous_brain_cons0a/results/cls0b_final/seed3_bank.pt
r=$(FAKE_MANIFEST=ok remote); [ "${r%% *}" != 0 ] && [ ! -e $T/ws/cons0a.log ] || fail "corrupt bank not stopped ('$r')"

# 8b stream tokenizer missing -> STOP
setup; rm $T/ws/Continuous_brain/models/qwen3-0.6b/tokenizer.json
r=$(FAKE_MANIFEST=ok remote); [ "${r%% *}" != 0 ] && grep -q 'stream tokenizer' $T/pre.txt || fail "missing stream tokenizer ('$r')"

# 9 PC-side tail: verified download, exit mapping 0/4/1, corrupt bundle rejected
mkdir -p $T/pc/ws $T/pc/dl $T/pc/home
printf '#!/bin/sh\nexit 0\n' > $T/bin/runpodctl; chmod +x $T/bin/runpodctl
printf '#!/usr/bin/env bash\nbash -c "$(printf %%s "$1" | sed "s|/workspace|%s|g")"\n' "$T/pc/ws" > $T/fakessh; chmod +x $T/fakessh
head=$(grep -n '^D=/workspace' "$L" | cut -d: -f1); start=$(grep -n '^mkdir -p \$OUT' "$L" | cut -d: -f1)
{ echo 'set -u'; sed -n "${head}p" "$L"; echo 'S="$FAKESSH"; POD=fakepod; DONE=0'; sed -n "${start},\$p" "$L"; } \
  | sed "s|/mnt/c/Users/davei/Downloads|$T/pc/dl|g" > $T/tail.sh
tail_run() { rm -rf $T/pc/ws/* $T/pc/home/cb_results
  echo payload > $T/pc/ws/x && (cd $T/pc/ws && tar czf results_cons0a.tar.gz x && sha256sum results_cons0a.tar.gz > results_cons0a.tar.gz.sha256)
  [ "${2:-0}" = 1 ] && echo corrupt >> $T/pc/ws/results_cons0a.tar.gz
  echo "CONS-0A JOB DONE exit=$1" > $T/pc/ws/cons0a.log
  HOME=$T/pc/home FAKESSH=$T/fakessh PATH=$T/bin:$PATH bash $T/tail.sh > $T/pre.txt 2>&1; echo $?; }
[ "$(tail_run 0)" = 0 ] && [ -f $T/pc/home/cb_results/cons0a/results_cons0a.tar.gz ] && [ -f $T/pc/dl/results_cons0a.tar.gz ] \
  || fail "PC tail success"
[ "$(tail_run 4)" = 4 ] || fail "PC tail: calibration STOP not mapped to 4"
[ "$(tail_run 3)" = 1 ] || fail "PC tail: failure not mapped to 1"
[ "$(tail_run 0 1)" = 1 ] || fail "PC tail: corrupted bundle accepted"
echo "PASS: cb_cons0a_run.sh (success, relaunch skip, contract STOP, projection STOP, seed failure + resume, model mismatch, lock, corrupt bank, stream tokenizer, PC tail)"
