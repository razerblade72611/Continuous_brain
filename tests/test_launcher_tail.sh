#!/usr/bin/env bash
# Regression check (Sol): a launcher's PC-side tail - download, checksum verification, Downloads copy, pod removal and
# exit-status propagation - must use ONE consistent output directory. Runs the real tail in a sandbox with a fake ssh
# (executes the remote command locally against a fake /workspace) and a fake runpodctl.
#   bash tests/test_launcher_tail.sh scripts/cb_cls0c_bs_run.sh results_cls0c_bs cls0c_bs.log "CLS-0C-BS JOB DONE"
set -uo pipefail
L=$1; BUNDLE=$2; LOG=$3; MARK=$4
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
mkdir -p $T/ws $T/dl $T/home $T/bin
printf '#!/bin/sh\nexit 0\n' > $T/bin/runpodctl; chmod +x $T/bin/runpodctl
cat > $T/fakessh <<SH
#!/usr/bin/env bash
bash -c "\$(printf '%s' "\$1" | sed "s|/workspace|$T/ws|g")"
SH
chmod +x $T/fakessh
head=$(grep -n '^D=/workspace' "$L" | cut -d: -f1)                       # variable line (D, OUT, DL ...)
start=$(grep -n '^mkdir -p ' "$L" | tail -n 1 | cut -d: -f1)              # the PC-side download tail
{ echo 'set -u'; sed -n "${head}p" "$L"; echo 'S="$FAKESSH"; POD=fakepod; DONE=0'; sed -n "${start},\$p" "$L"; } \
  | sed "s|/mnt/c/Users/davei/Downloads|$T/dl|g" > $T/tail.sh
run() {   # $1 = job exit code written to the log, $2 = 1 to corrupt the checksum
  rm -rf $T/ws/* $T/home/cb_results $T/dl/*
  echo payload > $T/ws/x && (cd $T/ws && tar czf $BUNDLE.tar.gz x && sha256sum $BUNDLE.tar.gz > $BUNDLE.tar.gz.sha256)
  [ "${2:-0}" = 1 ] && echo corrupt >> $T/ws/$BUNDLE.tar.gz
  echo "$MARK exit=$1" > $T/ws/$LOG
  HOME=$T/home FAKESSH=$T/fakessh PATH=$T/bin:$PATH bash $T/tail.sh > $T/out.txt 2>&1; echo $?
}
fail() { echo "FAIL ($L): $*"; cat $T/out.txt; exit 1; }
[ "$(run 0)" = 0 ] || fail "complete run did not exit 0"
ls $T/home/cb_results/*/$BUNDLE.tar.gz >/dev/null 2>&1 || fail "bundle not in the output directory"
[ $(ls -d $T/home/cb_results/* | wc -l) = 1 ] || fail "more than one output directory"
[ -f $T/dl/$BUNDLE.tar.gz ] || fail "no Downloads copy"
[ "$(run 4)" = 4 ] || fail "preflight-stop exit not propagated"
[ "$(run 0 1)" = 1 ] || fail "corrupted bundle not rejected"
echo "PASS: $L tail (download, verify, Downloads copy, exit propagation)"
