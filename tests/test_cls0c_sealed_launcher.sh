#!/usr/bin/env bash
# Regression test (Sol, 2026-10-30) for scripts/cb_cls0c_sealed_run.sh. Runs the REAL remote preflight + job script against a
# local clone of this repo in a sandbox ("/workspace" -> temp dir). Only the GPU stages (cls0c_sealed.py bridge-write / run),
# findmnt and pgrep are stubbed. Also runs the real PC-side download tail with a fake ssh/runpodctl.
#   bash tests/test_cls0c_sealed_launcher.sh
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd); L=$REPO/scripts/cb_cls0c_sealed_run.sh
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
PIN=$(grep -o '^PIN=[0-9a-f]*' "$L" | cut -d= -f2); FREEZE=$(grep -o '^FREEZE=[0-9a-f]*' "$L" | cut -d= -f2)
mkdir -p $T/bin
printf '#!/bin/sh\necho "srv:/networkvolumes/$VOL"\n' > $T/bin/findmnt; printf '#!/bin/sh\nexit 1\n' > $T/bin/pgrep
chmod +x $T/bin/*
awk '/bash -s" <<'"'"'REMOTE'"'"'/{f=1;next} /^REMOTE$/{f=0} f' "$L" > $T/remote.raw
[ -s $T/remote.raw ] || { echo "FAIL: could not extract the remote script"; exit 1; }
git clone -q "file://$REPO" $T/proto && git -C $T/proto checkout -q being/v3-bind 2>/dev/null
cat > $T/fakepy <<'PY'
#!/usr/bin/env bash
if [ "${1:-}" = "-" ]; then
  if [ "${FAKE_MANIFEST:-real}" = ok ]; then cat >/dev/null; echo "MODEL MANIFESTS OK"; exit 0; fi
  exec python3 -
fi
out=; prev=; for x in "$@"; do [ "$prev" = --out ] && out=$x; prev=$x; done
case "$*" in
  *"cls0c_sealed.py bridge-write"*) mkdir -p "$out"; [ "${BRIDGE_EXIT:-0}" = 0 ] && echo '{}' > "$out/bridge_seed101_recon.json"
     echo "BRIDGE WRITE DONE (stub)"; exit "${BRIDGE_EXIT:-0}";;
  *"cls0c_sealed.py run"*) mkdir -p "$out"; touch "$out/RUN_STARTED"; echo "CLS-0C SEALED DONE (stub)"; exit "${RUN_EXIT:-0}";;
esac
exec python3 "$@"
PY
chmod +x $T/fakepy
setup() {   # fresh sandbox volume with the clone, a fake venv and fake model dirs
  rm -rf $T/ws; mkdir -p $T/ws/Continuous_brain/.venv/bin $T/ws/hf/Qwen3-8B $T/ws/hf/Qwen3-Embedding-0.6B
  cp -a $T/proto $T/ws/Continuous_brain_c0csealed
  cp $T/fakepy $T/ws/Continuous_brain/.venv/bin/python
  echo '{}' > $T/ws/hf/Qwen3-8B/config.json; echo '{}' > $T/ws/hf/Qwen3-Embedding-0.6B/config.json
  sed "s|/workspace|$T/ws|g" $T/remote.raw > $T/remote.sh
}
remote() {  # runs preflight (+ job launch); waits for the job; prints "preflight_exit job_exit"
  VOL=29qhoqghqf D=$T/ws/Continuous_brain_c0csealed PIN=$PIN FREEZE=$FREEZE CB_UNLOCK=${CB_UNLOCK:-0} PATH=$T/bin:$PATH \
    bash $T/remote.sh > $T/pre.txt 2>&1; local pe=$?
  local je=none
  if [ $pe = 0 ]; then
    for _ in $(seq 60); do grep -q 'CLS-0C-SEALED JOB DONE' $T/ws/cls0c_sealed.log 2>/dev/null && break; sleep 0.5; done
    je=$(grep -o 'CLS-0C-SEALED JOB DONE exit=[0-9]*' $T/ws/cls0c_sealed.log | cut -d= -f2)
  fi
  echo "$pe $je"
}
fail() { echo "FAIL: $*"; echo "--- preflight"; cat $T/pre.txt; echo "--- job log"; cat $T/ws/cls0c_sealed.log 2>/dev/null; exit 1; }
R=$T/ws/Continuous_brain_c0csealed/results/cls0c_sealed_run

# 1 success: preflight passes, bridge then run, bundle written, lock released
setup; r=$(FAKE_MANIFEST=ok remote); [ "$r" = "0 0" ] || fail "success path gave '$r'"
grep -q 'PREFLIGHT OK' $T/pre.txt || fail "no PREFLIGHT OK"
[ -f $R/RUN_STARTED ] || fail "stage 2 did not run"
[ -f $T/ws/results_cls0c_sealed.tar.gz ] && (cd $T/ws && sha256sum -c --quiet results_cls0c_sealed.tar.gz.sha256) || fail "bundle missing/bad"
[ ! -e $T/ws/cls0c_sealed.lock ] || fail "lock not released"
grep -q 'stage 1' $T/ws/cls0c_sealed.log && grep -q 'stage 2' $T/ws/cls0c_sealed.log || fail "stages not logged"
# 2 a second launch after RUN_STARTED is refused in preflight (job never started)
r=$(FAKE_MANIFEST=ok remote); [ "${r%% *}" != 0 ] && grep -q 'RUN_STARTED exists' $T/pre.txt || fail "rerun not refused ('$r')"
# 3 bridge failure: stage 2 NOT started, exit propagated, bundle still written
setup; r=$(FAKE_MANIFEST=ok BRIDGE_EXIT=3 remote); [ "$r" = "0 3" ] || fail "bridge failure gave '$r'"
[ ! -e $R/RUN_STARTED ] || fail "stage 2 started after a failed bridge"
[ -f $T/ws/results_cls0c_sealed.tar.gz ] || fail "no bundle after bridge failure"
# 4 stage-2 failure exit propagated
setup; r=$(FAKE_MANIFEST=ok RUN_EXIT=2 remote); [ "$r" = "0 2" ] || fail "run failure gave '$r'"
# 5 corrupted sealed.json -> STOP before launch
setup; echo x >> $T/ws/Continuous_brain_c0csealed/results/cls0c_sealed/sealed.json
r=$(FAKE_MANIFEST=ok remote); [ "${r%% *}" != 0 ] && [ ! -e $T/ws/cls0c_sealed.log ] || fail "corrupt sealed.json not stopped ('$r')"
# 6 model files differing from the CLS-0B manifests (REAL manifest code on fake model dirs) -> STOP before launch
setup; r=$(remote); [ "${r%% *}" != 0 ] && grep -q 'MODEL MANIFESTS MISMATCH' $T/pre.txt && [ ! -e $T/ws/cls0c_sealed.log ] \
  || fail "model mismatch not stopped ('$r')"
# 7 lock held by another job -> STOP; CB_UNLOCK=1 clears it and proceeds
setup; mkdir $T/ws/cls0c_sealed.lock; echo "otherpod" > $T/ws/cls0c_sealed.lock/owner
r=$(FAKE_MANIFEST=ok remote); [ "${r%% *}" != 0 ] && grep -q 'lock held by otherpod' $T/pre.txt || fail "held lock not stopped ('$r')"
r=$(FAKE_MANIFEST=ok CB_UNLOCK=1 remote); [ "$r" = "0 0" ] || fail "CB_UNLOCK did not proceed ('$r')"
# 8 harness differing from the freeze -> STOP (committed change inside the sandbox clone)
setup; (cd $T/ws/Continuous_brain_c0csealed && git checkout -q --detach $PIN && echo '# x' >> scripts/cls0c_b.py \
  && git -c user.email=t@t -c user.name=t commit -qam x && git branch -qf being/v3-bind HEAD)
sed -i "s|git checkout -q --detach \$PIN|git checkout -q --detach HEAD; PIN=\$(git rev-parse HEAD)|" $T/remote.sh
r=$(FAKE_MANIFEST=ok remote); [ "${r%% *}" != 0 ] && grep -q 'harness differs' $T/pre.txt || fail "harness drift not stopped ('$r')"

# 9 PC-side tail: download/verify/exit mapping
mkdir -p $T/pc/ws $T/pc/dl $T/pc/home
printf '#!/bin/sh\nexit 0\n' > $T/bin/runpodctl; chmod +x $T/bin/runpodctl
printf '#!/usr/bin/env bash\nbash -c "$(printf %%s "$1" | sed "s|/workspace|%s|g")"\n' "$T/pc/ws" > $T/fakessh; chmod +x $T/fakessh
head=$(grep -n '^D=/workspace' "$L" | cut -d: -f1); start=$(grep -n '^mkdir -p \$OUT' "$L" | cut -d: -f1)
{ echo 'set -u'; sed -n "${head}p" "$L"; echo 'S="$FAKESSH"; POD=fakepod; DONE=0'; sed -n "${start},\$p" "$L"; } \
  | sed "s|/mnt/c/Users/davei/Downloads|$T/pc/dl|g" > $T/tail.sh
tail_run() { rm -rf $T/pc/ws/* $T/pc/home/cb_results
  echo payload > $T/pc/ws/x && (cd $T/pc/ws && tar czf results_cls0c_sealed.tar.gz x && sha256sum results_cls0c_sealed.tar.gz > results_cls0c_sealed.tar.gz.sha256)
  [ "${2:-0}" = 1 ] && echo corrupt >> $T/pc/ws/results_cls0c_sealed.tar.gz
  echo "CLS-0C-SEALED JOB DONE exit=$1" > $T/pc/ws/cls0c_sealed.log
  HOME=$T/pc/home FAKESSH=$T/fakessh PATH=$T/bin:$PATH bash $T/tail.sh > $T/pre.txt 2>&1; echo $?; }
[ "$(tail_run 0)" = 0 ] && [ -f $T/pc/home/cb_results/cls0c_sealed/results_cls0c_sealed.tar.gz ] && [ -f $T/pc/dl/results_cls0c_sealed.tar.gz ] \
  || fail "PC tail success path"
[ "$(tail_run 2)" = 1 ] || fail "PC tail: job failure not mapped to exit 1"
[ "$(tail_run 0 1)" = 1 ] || fail "PC tail: corrupted bundle accepted"
echo "PASS: cb_cls0c_sealed_run.sh (success, rerun refused, bridge failure, run failure, corrupt seal, model mismatch, lock, harness drift, PC tail)"
