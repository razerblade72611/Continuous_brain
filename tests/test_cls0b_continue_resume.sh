#!/usr/bin/env bash
# Regression test for the CLS-0B continuation orchestrator's progress-record protection (operations only).
# Two consecutive resumes of seed 3 with different progress records; the second worker stops at its preflight
# (exit 4, overwriting the record as the frozen runner does). The exact second record must be restored byte-for-byte,
# even when the older backup has a newer modification time (cp -p preserves mtimes).
#   bash tests/test_cls0b_continue_resume.sh [path/to/cb_cls0b_continue.sh]
set -euo pipefail
SRC=${1:-$(dirname "$0")/../scripts/cb_cls0b_continue.sh}
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
W=$T/workspace; R=$W/Continuous_brain_bind
mkdir -p $R/results/cls0b $R/scripts $R/.venv/bin $T/bin
# the remote orchestrator exactly as the launcher writes it, with /workspace pointed at the sandbox
sed -n "/^cat > \/workspace\/cls0b_continue.sh <<'ORCH'$/,/^ORCH$/p" "$SRC" | sed '1d;$d' | sed "s|/workspace|$W|g" > $T/orch.sh
grep -q 'for s in 3 4 5' $T/orch.sh || { echo "FAIL: could not extract orchestrator"; exit 1; }
ln -s "$(command -v python3)" $R/.venv/bin/python
printf '#!/bin/sh\necho "0 MiB, 0 MiB"\n' > $T/bin/nvidia-smi; chmod +x $T/bin/nvidia-smi
# the RunPod network volume refuses permission preservation: `cp -p` copies the content but exits non-zero
cat > $T/bin/cp <<SH
#!/usr/bin/env bash
p=0; a=(); for x in "\$@"; do case "\$x" in -p|--preserve*) p=1;; *) a+=("\$x");; esac; done
if [ \$p = 1 ]; then $(command -v cp) --preserve=timestamps "\${a[@]}" || exit 1; else $(command -v cp) "\${a[@]}" || exit 1; fi
[ \$p = 1 ] && { echo "cp: preserving permissions: Operation not permitted" >&2; exit 1; }; exit 0
SH
chmod +x $T/bin/cp
# fake worker: FAKE=timeout appends a write and is killed by the deadline; FAKE=preflight overwrites the record and exits 4
cat > $R/scripts/cls0b.py <<'PY'
import json, os, sys
out = sys.argv[sys.argv.index("--out") + 1]
if os.environ["FAKE"] == "timeout":
    d = json.load(open(out)); d["writes"]["e2"] = {"ok": True}; json.dump(d, open(out, "w")); sys.exit(124)
json.dump({"status": "preflight_stop", "writes": {}}, open(out, "w")); sys.exit(4)
PY
put() { printf '%s' "$2" > $R/results/cls0b/seed3.json; }
for s in 2 4 5; do echo '{"status":"complete","writes":{},"episodes":[],"checkpoints":[]}' > $R/results/cls0b/seed$s.json; done
run() { (cd $R && PATH=$T/bin:$PATH FAKE=$1 bash $T/orch.sh $(( $(date +%s) + 3600 ))) > $T/run_$1.log 2>&1 || true; }

put 3 '{"status": "writing", "writes": {"e1": {"ok": true}}}'
touch -d '2030-01-01' $R/results/cls0b/seed3.json            # first record: NEWER mtime (worst case for mtime-based selection)
run timeout                                                  # resume 1: progresses the record, then interrupted
SECOND=$(cat $R/results/cls0b/seed3.json)
grep -q '"e2"' <<< "$SECOND" || { echo "FAIL: resume 1 did not progress the record"; cat $T/run_timeout.log; exit 1; }
touch -d '2020-01-01' $R/results/cls0b/seed3.json            # second record: OLDER mtime
sleep 1                                                      # distinct backup timestamps
run preflight                                                # resume 2: worker stops at its preflight
grep -q 'status=CLS0B_PREFLIGHT_STOP' $T/run_preflight.log || { echo "FAIL: status"; cat $T/run_preflight.log; exit 1; }
[ "$(cat $R/results/cls0b/seed3.json)" = "$SECOND" ] || { echo "FAIL: restored record is not the exact second record"; \
  cat $R/results/cls0b/seed3.json; echo; cat $T/run_preflight.log; exit 1; }
ls $R/results/cls0b/seed3.json.preflight_stop.* >/dev/null || { echo "FAIL: overwritten record not kept"; exit 1; }
[ $(ls $R/results/cls0b/seed3.json.pre_run_backup.* | wc -l) -eq 2 ] || { echo "FAIL: expected two backups"; exit 1; }
echo "PASS: exact second progress record restored byte-for-byte after a preflight stop"
