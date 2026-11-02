#!/usr/bin/env bash
# SUPPORT-1 launchers (phase 1 and phase 2): run the reviewed ABST-0 launcher regressions (boot polling, cleanup and
# preservation, job upload / exactly-once start / poll) against cb_support1_p{1,2}.sh, with names mapped
# support1 -> abst0 so the reviewed tests run unmodified in logic.  bash tests/test_support1_launcher.sh
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd); T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
mkdir -p $T/scripts $T/tests
for ph in 1 2; do
  cp $REPO/scripts/cb_support1_p$ph.sh $T/scripts/cb_abst0_p$ph.sh
  sed -i "s/support1_p$ph/abst0_p$ph/g; s/Continuous_brain_support1/Continuous_brain_abst0/g; s/SUPPORT1 P$ph DONE/ABST0 P$ph DONE/g; s#/workspace/support1#/workspace/abst0#g; s/support1_ckpt/abst0_ckpt/g; s/support1_results/abst0_results/g; s/find support1 /find abst0 /g; s/--exclude=.\*\.pt. support1/--exclude='*.pt' abst0/g" $T/scripts/cb_abst0_p$ph.sh
done
for t in boot cleanup job; do
  sed -e '/phase-1 PIN unresolved/d' -e 's/x 2 launchers/x SUPPORT-1 launchers (phase 1, phase 2)/' $REPO/tests/test_abst0_$t.sh > $T/tests/t_$t.sh
  sed -i "s#^REPO=\$(cd \"\$(dirname \"\$0\")/..\" \&\& pwd);#REPO=$T;#" $T/tests/t_$t.sh
  grep -q "^REPO=$T; fail()" $T/tests/t_$t.sh || { echo "FAIL: harness could not retarget REPO in $t (fail() must survive)"; exit 1; }
  out=$(bash $T/tests/t_$t.sh 2>&1); rc=$?; echo "$out" | tail -1
  [ $rc = 0 ] && ! grep -q "command not found" <<< "$out" || { echo "FAIL: $t regression failed (rc=$rc)"; exit 1; }
done
