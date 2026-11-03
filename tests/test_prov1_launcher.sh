#!/usr/bin/env bash
# PROV-1 launcher safety regressions (scripts/cb_prov1.sh): preserve() and cleanup() extracted verbatim and run against a
# fake pod (ssh mocked by a local shell over a fake /workspace) and a mocked runpodctl.
#   - evidence + phase B: results bundle and every checkpoint streamed, sha256-verified, THEN the pod is removed;
#   - a corrupted checkpoint stream: verification fails, the pod is STOPPED and kept, never removed;
#   - a corrupted results stream: same;
#   - no evidence on the pod: removed;  unreachable pod (ssh fails): STOPPED and kept;
#   - phase A (KEEP_CKPT=0): checkpoint hashes recorded, no checkpoint files copied.
#   - argument and PIN guards refuse to start.
#   bash tests/test_prov1_launcher.sh
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd); T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
L=${PROV1_LAUNCHER:-$REPO/scripts/cb_prov1.sh}
fail() { echo "FAIL: $*"; exit 1; }
printf 'import sys\nd = bytearray(sys.stdin.buffer.read())\nd[min(3000, len(d) - 1)] ^= 0xFF\nsys.stdout.buffer.write(d)\n' > $T/flip.py   # 1 byte flipped: tar still accepts the stream
FN=$T/fn.sh
awk '/^preserve\(\) \{/,/^}/' $L > $FN; awk '/^cleanup\(\) \{/,/^}/' $L >> $FN
grep -q '^preserve() {' $FN && grep -q '^cleanup() {' $FN || fail "could not extract preserve/cleanup"

setup() {  # $1 = evidence kind: ckpt | none
  rm -rf $T/ws $T/out $T/calls; mkdir -p $T/ws $T/out; : > $T/calls
  if [ "$1" = ckpt ]; then
    mkdir -p $T/ws/prov1/seed20; echo '{"x":1}' > $T/ws/prov1/seed20/eval_RC_s1_merged.json
    head -c 200000 /dev/urandom > $T/ws/prov1/seed20/cortex_RC_s1.pt; head -c 150000 /dev/urandom > $T/ws/prov1/seed20/cortex_RU_s4_withheld.pt
  fi
}
run() {  # $1 KEEP_CKPT, $2 ssh mode: ok | down | corrupt_ckpt | corrupt_results
  cat > $T/run.sh <<EOF
set -uo pipefail
OUT=$T/out; KEEP_CKPT=$1; POD=podX; DONE=0
runpodctl() { echo "runpodctl \$*" >> $T/calls; }
ssh_fake() {
  local c="\${*//\/workspace/$T/ws}"
  case "$2" in
    down) return 255;;
    corrupt_ckpt) if [[ "\$c" == *"tar cf - -T -"* ]]; then bash -c "\$c" | python3 $T/flip.py; return 0; fi;;   # 1 byte flipped: tar accepts it
    corrupt_results) if [[ "\$c" == *"cat $T/ws/prov1_results.tar.gz" && "\$c" != *sha256* ]]; then bash -c "\$c" | python3 $T/flip.py; return 0; fi;;
  esac
  bash -c "\$c"
}
S=ssh_fake
$(cat $FN)
cleanup
EOF
  bash $T/run.sh > $T/log 2>&1
}
# 1. evidence, phase B: preserved and verified, then removed
setup ckpt; run 1 ok
grep -q "preserved + verified" $T/log || fail "phase B preserve did not verify: $(cat $T/log)"
grep -q "pod remove podX" $T/calls && ! grep -q "pod stop" $T/calls || fail "phase B: pod not removed after verified preserve"
cmp -s $T/out/ckpt/prov1/seed20/cortex_RC_s1.pt $T/ws/prov1/seed20/cortex_RC_s1.pt || fail "checkpoint not copied exactly"
[ "$(wc -l < $T/out/prov1_ckpt.sha256)" = 2 ] || fail "checkpoint hash list incomplete"
(cd $T/out && sha256sum -c --quiet prov1_results.tar.gz.sha256) || fail "results bundle not verifiable"
echo "ok 1 evidence (phase B): streamed, verified, removed"
# 2. corrupted checkpoint stream: kept
setup ckpt; run 1 corrupt_ckpt
grep -q "pod stop podX" $T/calls && ! grep -q "pod remove" $T/calls || fail "corrupt checkpoint stream: pod was not kept ($(cat $T/calls))"
echo "ok 2 corrupted checkpoint stream: pod STOPPED and kept"
# 3. corrupted results stream: kept
setup ckpt; run 1 corrupt_results
grep -q "pod stop podX" $T/calls && ! grep -q "pod remove" $T/calls || fail "corrupt results stream: pod was not kept"
echo "ok 3 corrupted results stream: pod STOPPED and kept"
# 4. no evidence: removed
setup none; run 1 ok
grep -q "confirmed no evidence" $T/log && grep -q "pod remove podX" $T/calls || fail "no evidence: pod not removed"
echo "ok 4 no evidence: removed"
# 5. unreachable pod: kept
setup ckpt; run 1 down
grep -q "pod stop podX" $T/calls && ! grep -q "pod remove" $T/calls || fail "unreachable pod: not kept"
echo "ok 5 unreachable pod: STOPPED and kept"
# 6. phase A: hashes recorded, no checkpoint files copied, removed
setup ckpt; run 0 ok
[ ! -e $T/out/ckpt ] && [ "$(wc -l < $T/out/prov1_ckpt.sha256)" = 2 ] && grep -q "pod remove podX" $T/calls || fail "phase A preserve"
echo "ok 6 phase A: hashes only, removed"
# 7. guards
o=$(bash $L 2>&1); [[ "$o" == *usage* ]] || fail "no-argument guard"
o=$(bash $L 1 2>&1); [[ "$o" == *"PIN is not a resolved commit hash"* ]] || fail "PIN guard (UNPINNED must refuse)"
echo "ok 7 argument and PIN guards"
echo "PROV-1 launcher: 7/7 passed"
