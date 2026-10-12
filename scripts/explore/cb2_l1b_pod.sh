#!/usr/bin/env bash
# CB2-L1b (DAV-99), pod side.  Three phases, each ending in an explicit stop; nothing proceeds to the next phase by
# itself (docs/explore/CB2_L1B_PREREG_DRAFT.md s7-8, pod-runbook.md):
#   PHASE=bench  GPU rehearsal on the real hardware (smoke blocks, no claim) + bench-sleep (both sleeps as the run will
#                run them) + bench-lives (S2b and a learner with its chooser active, two worker counts).  Prints the
#                measured projection, packs it, STOPS.  Price from these numbers, then decide.
#   PHASE=run    the locked 10-cycle loop (resumable only as the same locked experiment), loop report, pack, STOP.
#   PHASE=final  the final battery + report + Gate 4 probes, pack.  Can run on a CPU pod holding the run directory.
# Launch:  PHASE=bench setsid nohup bash /workspace/cb2_l1b_pod.sh > /workspace/l1b-bench.log 2>&1 < /dev/null &
# Needs in /workspace: tile_decoder_v1.pt, s2b_min.tgz, cb2_kit_r2.tgz, cb2_l1b_code.bundle + cb2_l1b_code.rev,
# cb2_l1b_SHA256SUMS (and, for PHASE=final on a fresh pod, cb2_l1b_loop.tgz from PHASE=run).
set -euo pipefail
PHASE=${PHASE:?set PHASE=bench, run or final}
W=/workspace; CODE=$W/Continuous_brain; BR=explore/cb2-l1fix
OUT=runs/explore/cb2/l1b; REH=runs/explore/cb2/l1b-rehearsal
cd $W
echo "== $(date -u) l1b $PHASE: verifying inputs"
sha256sum -c <<SUMS
0c46b2d1ace00c027d45e23ae01b00da8b802d10660122da49f9727a1cfe9999  tile_decoder_v1.pt
81e31395144364bcc55b4f57ac97cb8cdc94947e5c22730a4e584d6b8603c5f8  s2b_min.tgz
2d3ec50ce05e635a46fc2d658a2395aa65de929088c17879090e751f15a1a1ec  cb2_kit_r2.tgz
SUMS
sha256sum -c cb2_l1b_SHA256SUMS
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "(no GPU)"
echo "cpus (host): $(nproc)  mem: $(free -g | awk '/Mem:/{print $2" GB"}')"
if [ ! -f $W/.l1b_restored ]; then
  echo "== restoring code, vision v1, S2b, the lived archive"
  if [ ! -d "$CODE/.git" ]; then
    mkdir -p "$CODE" && git -C "$CODE" init -q && git -C "$CODE" fetch -q "$W/cb2_l1b_code.bundle" $BR
    git -C "$CODE" checkout -q --detach FETCH_HEAD
  fi
  cd "$CODE"
  mkdir -p runs/explore/vision runs/explore/cb2/bootstrap
  cp "$W/tile_decoder_v1.pt" runs/explore/vision/
  tar --no-same-owner -C "$CODE" -xzf $W/s2b_min.tgz
  mkdir -p $W/kit && tar --no-same-owner -C $W/kit -xzf $W/cb2_kit_r2.tgz ./cb2/archive
  mv $W/kit/cb2/archive/lived-cb2-ind-0001 runs/explore/cb2/bootstrap/
  touch $W/.l1b_restored
fi
cd "$CODE"
# runbook 9: nothing from an earlier run may still be alive, and no GPU memory may be in use
if pgrep -f 'cb2_l1\.py|multiprocessing|/workspace/venv/bin/python' >/dev/null; then
  echo "STOP: processes from an earlier run are still alive:"; ps -eo pid,etime,cmd | grep -E 'python|multiprocessing' | grep -v grep | cut -c1-150; exit 1
fi
if nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '$1 > 200 {b=1} END {exit !b}'; then
  echo "STOP: GPU memory is in use by something else:"; nvidia-smi; exit 1
fi
# one code revision: a reused pod gets the uploaded code; the run's own lock refuses any other revision on resume
if [ "$(git rev-parse HEAD)" != "$(cat $W/cb2_l1b_code.rev)" ]; then
  git fetch -q "$W/cb2_l1b_code.bundle" $BR && git checkout -q --detach FETCH_HEAD
fi
[ "$(git rev-parse HEAD)" = "$(cat $W/cb2_l1b_code.rev)" ] || { echo "STOP: code revision differs"; exit 1; }
[ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "STOP: tracked files modified"; exit 1; }
# shellcheck source=cb2_l1b_pack.sh
. scripts/explore/cb2_l1b_pack.sh                    # packing: every needed log travels inside the run directory
PINS="numpy==2.4.4 scipy==1.17.1 crafter==1.8.3 opensimplex==0.4.5.1 pillow==12.2.0 pyarrow==25.0.1 zstandard==0.25.0 jsonschema==4.26.0 pytest"
if [ ! -x $W/venv/bin/python ]; then
  if python3 -m venv --system-site-packages $W/venv 2>/dev/null; then $W/venv/bin/pip install -q $PINS
  else mkdir -p $W/venv/bin && ln -sf "$(command -v python3)" $W/venv/bin/python
       python3 -m pip install -q $PINS || python3 -m pip install -q --break-system-packages $PINS; fi
fi
export PATH=$W/venv/bin:$PATH PYTHONPATH=$CODE OPENBLAS_NUM_THREADS=1
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -m pytest -q -p no:cacheprovider tests/test_cb2_core.py tests/test_cb2_core_map.py tests/test_cb2_consolidation.py \
  tests/test_cb2_chooser_guards.py tests/test_cb2_l1_repairs.py tests/test_cb2_l1b.py tests/test_cb2_view_geometry.py
python - <<'PY'
import json
m = json.load(open("runs/explore/cb2/candidates/s2b-final/cb2-ind-0001/individual.json"))
ok = m["state_hash"].startswith("cc8948e7") and m["weights_sha256"].startswith("8cffde79")
print("S2b twin source:", "OK" if ok else "MISMATCH"); raise SystemExit(0 if ok else 1)
PY
# runbook 1: workers from the pod's real limits (cgroup CPU quota - 1, ~1.6 GB per worker), never nproc
set +e +o pipefail
CG=${CG:-/sys/fs/cgroup}; N=$(nproc); Q=""
[ -r $CG/cpu.max ] && Q=$(awk '$1!="max"{print int($1/$2)}' $CG/cpu.max)
[ -z "$Q" ] && [ -r $CG/cpu/cpu.cfs_quota_us ] && Q=$(awk 'NR==FNR{q=$1;next} q>0{print int(q/$1)}' $CG/cpu/cpu.cfs_quota_us $CG/cpu/cpu.cfs_period_us)
if [ -n "$Q" ] && [ "$Q" -gt 0 ]; then [ "$Q" -lt "$N" ] && N=$Q; else N=$(( N < 12 ? N : 12 )); fi
M=""
for f in $CG/memory.max $CG/memory/memory.limit_in_bytes; do
  [ -z "$M" ] && [ -r $f ] && M=$(awk '$1!="max" && $1<1e15{print int($1/1717986918)}' $f)
done
[ -z "$M" ] && M=$(free -g | awk '/Mem:/{m=int($2/1.6); print (m < 19 ? m : 19)}')
NG=$(nvidia-smi -L 2>/dev/null | grep -c GPU)
set -euo pipefail
# sleeps take ~3 GB host RAM each (CPU smoke measurement): keep 8 GB of the memory slots for the two sleepers
SLOTS=$(( NG > 0 ? M - 5 : M )); SLOTS=$(( SLOTS < 1 ? 1 : SLOTS ))
WK=$(( N - 1 < SLOTS ? N - 1 : SLOTS )); WK=$(( WK < 1 ? 1 : WK )); WK=${WORKERS:-$WK}
[ "$NG" = "1" ] && export L1_PARALLEL_SLEEP=1          # one GPU: both twins sleep at once on it (as benchmarked)
export L1_CPU_QUOTA=$N                                 # python reads the same quota (cgroup v1 pods: see _cpu_quota)
echo "cpu quota: $N  memory slots: $M  gpus: $NG  workers: $WK  parallel-sleep-on-one-gpu: ${L1_PARALLEL_SLEEP:-0}"

case "$PHASE" in
bench)
  echo "== $(date -u) GPU rehearsal (smoke blocks, practice run 99: can never make a claim)"
  python scripts/explore/cb2_l1.py run --rehearsal --twin runs/explore/cb2/candidates/s2b-final --bootstrap runs/explore/cb2/bootstrap \
    --out $REH --cycles 1 --lives 4 --holdout-lives 1 --workers $WK --probe-worlds 4 --situation-worlds 2 \
    --core-updates 50 --anchor-updates 50 --chooser-updates 50 --final-worlds 4 --chooser-off-worlds 2 2>&1 | tee $W/l1b-rehearsal.log
  python scripts/explore/cb2_l1.py final --out $REH --workers $WK
  python scripts/explore/cb2_l1.py report --out $REH --report $REH/report.json | tee -a $W/l1b-rehearsal.log
  echo "== $(date -u) bench-sleep: both sleeps exactly as the run will run them"
  C=$([ "$NG" = "1" ] && echo 2 || echo 1)
  python scripts/explore/cb2_l1.py bench-sleep --twin runs/explore/cb2/candidates/s2b-final/cb2-ind-0001 \
    --own $REH/archive-learning --bootstrap runs/explore/cb2/bootstrap --out $W/bench-sleep --updates 100 --concurrent $C \
    | tee $W/l1b-bench-sleep.json
  echo "== $(date -u) bench-lives: S2b and the rehearsal's learner (chooser active), at two worker counts"
  python scripts/explore/cb2_l1.py bench-lives --twin runs/explore/cb2/candidates/s2b-final --workers $(( WK / 2 > 0 ? WK / 2 : 1 )),$WK --lives 60 | tee $W/l1b-bench-lives-s2b.json
  python scripts/explore/cb2_l1.py bench-lives --twin $REH/twins/learning --workers $WK --lives 60 | tee $W/l1b-bench-lives-learner.json
  tar -czf $W/cb2_l1b_bench.tgz $REH $W/l1b-*.json $W/l1b-rehearsal.log
  sha256sum $W/cb2_l1b_bench.tgz
  echo "== $(date -u) l1b BENCH finished -- STOP: price the run from these measurements before PHASE=run"
  ;;
run)
  echo "== $(date -u) l1b loop with $WK workers (locked at creation; resumes only as the same experiment)"
  RES=""; [ -f $OUT/run.json ] && RES="--resume"
  python scripts/explore/cb2_l1.py run --twin runs/explore/cb2/candidates/s2b-final --bootstrap runs/explore/cb2/bootstrap \
    --out $OUT --workers $WK $RES 2>&1 | grep --line-buffered -v '^{$\|^}$' | tee -a $W/l1b-run.log
  python scripts/explore/cb2_l1.py report --out $OUT --report $OUT/loop-report.json > /dev/null
  l1b_pack_loop $W "$CODE" $OUT
  echo "== $(date -u) l1b LOOP finished -- STOP: download + verify cb2_l1b_loop.tgz, then PHASE=final (here or a CPU pod)"
  ;;
final)
  l1b_restore_loop $W "$CODE" $OUT
  echo "== $(date -u) l1b final battery with $WK workers"
  python scripts/explore/cb2_l1.py final --out $OUT --workers $WK 2>&1 | tee -a $W/l1b-final.log
  python scripts/explore/cb2_l1.py report --out $OUT --report evidence/explore/cb2/l1b-report.json | tee -a $W/l1b-final.log
  echo "== $(date -u) gate 4 probes (secondary): each twin's core on the learning twin's held-out lives, corrected [x,y] mask"
  python - <<'PY'
import io, json, torch
from pathlib import Path
for t in ("learning", "yoked", "frozen"):
    h = Path(f"runs/explore/cb2/l1b/twins/{t}/cb2-ind-0001"); m = json.loads((h / "individual.json").read_text())
    s = torch.load(io.BytesIO((h / "weights" / f"{m['weights_sha256']}.pt").read_bytes()), map_location="cpu", weights_only=False)
    torch.save(s["core"], f"runs/explore/cb2/l1b/core-{t}.pt")
PY
  for t in learning yoked frozen; do
    python scripts/explore/cb2_sleep.py probe-life --archive $OUT/archive-learning --heldout-lives 20 \
      --test-archive $OUT/archive-heldout --core $OUT/core-$t.pt --fit-lives 120 --test-lives 20 \
      --out evidence/explore/cb2/l1b-gate4-$t.json > /dev/null || echo "gate 4 probe for $t FAILED (secondary; reported)"
  done
  l1b_pack_final $W "$CODE" $OUT
  echo "== $(date -u) l1b FINAL finished"
  ;;
*) echo "unknown PHASE $PHASE"; exit 1 ;;
esac
