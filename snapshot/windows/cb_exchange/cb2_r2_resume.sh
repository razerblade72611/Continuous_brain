#!/usr/bin/env bash
# Resume CB2 round r2 on a fresh GPU pod whose /workspace is the network volume. Run ON THE POD:
#   setsid nohup bash /workspace/cb2_r2_resume.sh > /workspace/r2_resume.log 2>&1 < /dev/null &
# Restores the verified backups, applies the dream fix, regenerates practice run 2 exactly (WORKERS=24, same seeds),
# skips everything already trained (core, foresight gate, P-A, P-B), finishes dream -> probe -> candidate -> 40-world eval,
# then checks the regenerated practice data file-by-file against the original manifests. Safe to re-run.
set -euo pipefail
W=/workspace; KIT=$W/kit; CODE=$W/Continuous_brain
cd $W
echo "== $(date -u) r2 resume: verifying inputs"
sha256sum -c <<SUMS
2d3ec50ce05e635a46fc2d658a2395aa65de929088c17879090e751f15a1a1ec  cb2_kit_r2.tgz
d6874bf6d1da22d100f1dae12446317f2bca11c9ea232a860b7c4998389f2ef8  cb2_partial_r2.tgz
cc5da7cb2c8a4d8610472c641e2aca3456936a0e8fb07934e9736bba2b0e5ee8  cb2_fix_r2_dream.py
390722b2a0a15249472f71430a9598c90c0840f3d58134fdaf11e956e63cf2aa  practice_r2_manifests.tgz
SUMS
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "cpu quota: $(cat /sys/fs/cgroup/cpu.max 2>/dev/null)  mem limit: $(cat /sys/fs/cgroup/memory.max 2>/dev/null)"

if [ ! -f $W/.r2_restored ]; then
  echo "== restoring kit and code"
  mkdir -p $KIT && tar --no-same-owner -C $KIT -xzf cb2_kit_r2.tgz
  if [ ! -d "$CODE/.git" ]; then                       # code first: the backup holds a tracked file
    mkdir -p "$CODE" && git -C "$CODE" init -q && git -C "$CODE" fetch -q "$KIT/code.bundle" HEAD
    git -C "$CODE" checkout -q --detach FETCH_HEAD
  fi
  [ "$(git -C "$CODE" rev-parse HEAD)" = "$(cat $KIT/code.rev)" ] || { echo "STOP: code revision differs from the kit"; exit 1; }
  echo "== restoring trained r2 state (core, policies, gate, logs)"
  tar --no-same-owner -C $W -xzf cb2_partial_r2.tgz \
      --exclude='Continuous_brain/evidence/explore/cb2/bench_container.json' \
      --exclude='Continuous_brain/runs/explore/cb2/logs/r2/practice.done'
  mv $W/r2.log $W/r2_attempt1.log                      # the first attempt's full log, kept
  cp -r $CODE/runs/explore/cb2/logs/r2 $CODE/runs/explore/cb2/logs/r2_attempt1
  cd $CODE
  for f in core/core.pt policy-P-A/policy.pt policy-P-B/policy.pt; do
    exp=$(cut -d' ' -f1 runs/explore/cb2/sleep/r2/$f.sha256); act=$(sha256sum runs/explore/cb2/sleep/r2/$f | cut -d' ' -f1)
    [ "$exp" = "$act" ] && echo "$f: OK" || { echo "STOP: $f does not match its recorded hash"; exit 1; }
  done
  echo "== applying the dream fix"
  PYTHONPATH=$CODE python3 $W/cb2_fix_r2_dream.py
  touch $W/.r2_restored
fi

echo "== $(date -u) running the round (finished steps are skipped)"
cd $W
WORKERS=24 bash $KIT/cb2_pod_run.sh r2

echo "== $(date -u) checking regenerated practice run 2 against the original manifests"
rm -rf $W/practice_r2_orig && mkdir -p $W/practice_r2_orig && tar -C $W/practice_r2_orig -xzf $W/practice_r2_manifests.tgz
python3 - <<'PY'
import json, pathlib
orig = pathlib.Path("/workspace/practice_r2_orig"); new = pathlib.Path("/workspace/Continuous_brain")
same = diff = missing = 0
for m in sorted(orig.rglob("*.json")):
    n = new / m.relative_to(orig)
    if not n.exists(): missing += 1; print("MISSING", n); continue
    a, b = json.loads(m.read_text()), json.loads(n.read_text())
    fa = {k: v.get("sha256") for k, v in a.get("files", {}).items()}
    fb = {k: v.get("sha256") for k, v in b.get("files", {}).items()}
    if fa == fb and a.get("lives") == b.get("lives"): same += 1
    else: diff += 1; print("DIFFERS", m.relative_to(orig))
print(json.dumps({"practice_manifests_identical": same, "differ": diff, "missing": missing,
                  "verdict": "PRACTICE REGENERATED EXACTLY" if diff == missing == 0 else "PRACTICE NOT IDENTICAL - probes used regenerated data"}))
PY
echo "== $(date -u) packing what the results file leaves out: evaluation lives (next DAgger round's data) and dreams"
{ cd $CODE && tar -czf $W/cb2_r2_archive_extra.tgz runs/explore/cb2/archive/eval-r2-PA* runs/explore/cb2/archive/dream-r2 && \
  ls -la $W/cb2_r2_archive_extra.tgz && sha256sum $W/cb2_r2_archive_extra.tgz $W/cb2_results_r2.tgz; } || \
  echo "WARNING: packing the evaluation/dream archive failed - the pod will NOT be deleted until this is resolved"
echo "== $(date -u) r2 resume finished"
