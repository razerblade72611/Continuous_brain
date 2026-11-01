#!/usr/bin/env bash
# Regression for the ABST-0 launchers' cleanup(): the pod is removed ONLY after a confirmed "no evidence" or a verified
# preservation; ssh failure, an uninspectable directory or a corrupted download -> stop and KEEP.  bash tests/test_abst0_cleanup.sh
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd); fail() { echo "FAIL: $*"; exit 1; }
for L in scripts/cb_abst0_p1.sh scripts/cb_abst0_p2.sh; do
  T=$(mktemp -d); trap 'rm -rf "$T"' EXIT; mkdir -p $T/ws/abst0/seed12 $T/out $T/bin
  awk '/^preserve\(\) \{/{f=1} f{print} /^\}$/ && f && ++n==2{exit}' $REPO/$L | sed "s|/workspace|$T/ws|g" > $T/fns.sh
  printf '#!/bin/sh\necho "$*" >> %s/calls\n' "$T" > $T/bin/runpodctl; chmod +x $T/bin/runpodctl
  run() { rm -f $T/calls; (export PATH=$T/bin:$PATH; OUT=$T/out; POD=pod1; DONE=0; S="$T/fakessh"; source $T/fns.sh; cleanup) >/dev/null 2>&1; cat $T/calls; }
  ssh_ok() { printf '#!/usr/bin/env bash\nbash -c "$*"\n' > $T/fakessh; chmod +x $T/fakessh; }
  ssh_ok; echo pt > $T/ws/abst0/seed12/cortex_P_s1.pt
  [ "$(run)" = "pod remove pod1" ] && [ -f $T/out/ckpt/abst0/seed12/cortex_P_s1.pt ] || fail "$L: evidence + good download"
  printf '#!/usr/bin/env bash\nexit 255\n' > $T/fakessh
  [ "$(run)" = "pod stop pod1" ] || fail "$L: ssh failure must stop and keep"
  printf '#!/usr/bin/env bash\nbash -c "find() { return 1; }; export -f find; $*"\n' > $T/fakessh
  [ "$(run)" = "pod stop pod1" ] || fail "$L: find error must stop and keep"
  printf '#!/usr/bin/env bash\nif [[ "$*" == *"tar cf - abst0_ckpt.tar"* ]]; then bash -c "$*" | head -c 300; else bash -c "$*"; fi\n' > $T/fakessh
  rm -rf $T/out/*; [ "$(run)" = "pod stop pod1" ] || fail "$L: corrupted download must stop and keep"
  ssh_ok; rm -rf $T/ws/abst0; [ "$(run)" = "pod remove pod1" ] || fail "$L: confirmed no evidence -> remove"
  (export PATH=$T/bin:$PATH; OUT=$T/out; POD=pod1; DONE=0; S=""; source $T/fns.sh; rm -f $T/calls; cleanup >/dev/null 2>&1)
  [ "$(cat $T/calls)" = "pod remove pod1" ] || fail "$L: nothing ran (no ssh yet) -> remove"
  rm -rf "$T"
done
grep -q '^PIN=[0-9a-f]\{40\}$' $REPO/scripts/cb_abst0_p1.sh || fail "phase-1 PIN unresolved"
echo "PASS: ABST-0 launcher cleanup (evidence+verified, ssh failure, find error, corrupt download, no evidence, pre-ssh) x 2 launchers"
