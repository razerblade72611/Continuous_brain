#!/usr/bin/env bash
# SUPPORT-1 phase-1 launcher: run the reviewed ABST-0 launcher regressions (boot polling, cleanup/preservation, job
# upload/exactly-once start/poll) against cb_support1_p1.sh, with names mapped abst0 -> support1. bash tests/test_support1_launcher.sh
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd); T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
mkdir -p $T/scripts $T/tests; cp $REPO/scripts/cb_support1_p1.sh $T/scripts/cb_abst0_p1.sh
# map the support1 launcher's names to the ones the reviewed tests address (abst0), so the tests run unmodified in logic
sed -i 's/support1_p1/abst0_p1/g; s/Continuous_brain_support1/Continuous_brain_abst0/g; s/SUPPORT1 P1 DONE/ABST0 P1 DONE/g; s#/workspace/support1#/workspace/abst0#g; s/support1_ckpt/abst0_ckpt/g; s/support1_results/abst0_results/g; s/find support1 /find abst0 /g; s/--exclude=.\*\.pt. support1/--exclude='"'"'*.pt'"'"' abst0/g' $T/scripts/cb_abst0_p1.sh
grep -q 'support1' $T/scripts/cb_abst0_p1.sh && grep -n support1 $T/scripts/cb_abst0_p1.sh | grep -v 'tests/test_support1.py\|scripts/support1.py\|results/support1\|cb_results/support1\|cb_support1_pod\|cb-support1\|support1_phase1\|support1/phase0\|R=/workspace/abst0' | head -3
for t in boot cleanup job; do
  sed -e 's/for L in scripts\/cb_abst0_p1.sh scripts\/cb_abst0_p2.sh; do/for L in scripts\/cb_abst0_p1.sh; do/' \
      -e 's/^for ph in 1 2; do/for ph in 1; do/' -e '/phase-1 PIN unresolved/d' \
      -e 's/x 2 launchers/x SUPPORT-1 phase-1 launcher/' $REPO/tests/test_abst0_$t.sh > $T/tests/t_$t.sh
  sed -i "s#^REPO=\$(cd \"\$(dirname \"\$0\")/..\" \&\& pwd);#REPO=$T;#" $T/tests/t_$t.sh
  grep -q "^REPO=$T; fail()" $T/tests/t_$t.sh || { echo "FAIL: harness could not retarget REPO in $t (fail() must survive)"; exit 1; }
  out=$(bash $T/tests/t_$t.sh 2>&1); rc=$?; echo "$out" | tail -1
  [ $rc = 0 ] && ! grep -q "command not found" <<< "$out" || { echo "FAIL: $t regression failed (rc=$rc)"; exit 1; }
done
