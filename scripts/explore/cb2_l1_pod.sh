#!/usr/bin/env bash
# CB2-L1 (DAV-98), pod side: learning twin vs frozen twin, 10 wake/sleep cycles, probes every cycle.
#   setsid nohup bash /workspace/cb2_l1_pod.sh > /workspace/l1.log 2>&1 < /dev/null &
# Needs in /workspace: tile_decoder_v1.pt, s2b_min.tgz (the S2b individual), cb2_kit_r2.tgz (the canonical's lived
# archive before the split), cb2_l1_code.bundle + .rev + cb2_l1_SHA256SUMS.  Resumable (cycles.jsonl).
set -euo pipefail
W=/workspace; CODE=$W/Continuous_brain
cd $W
echo "== $(date -u) l1: verifying inputs"
sha256sum -c <<SUMS
0c46b2d1ace00c027d45e23ae01b00da8b802d10660122da49f9727a1cfe9999  tile_decoder_v1.pt
81e31395144364bcc55b4f57ac97cb8cdc94947e5c22730a4e584d6b8603c5f8  s2b_min.tgz
2d3ec50ce05e635a46fc2d658a2395aa65de929088c17879090e751f15a1a1ec  cb2_kit_r2.tgz
SUMS
sha256sum -c cb2_l1_SHA256SUMS
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "cpus: $(nproc)  mem: $(free -g | awk '/Mem:/{print $2" GB"}')"
if [ ! -f $W/.l1_restored ]; then
  echo "== restoring code, vision v1, S2b, the lived archive"
  if [ ! -d "$CODE/.git" ]; then
    mkdir -p "$CODE" && git -C "$CODE" init -q && git -C "$CODE" fetch -q "$W/cb2_l1_code.bundle" explore/cb2-r3
    git -C "$CODE" checkout -q --detach FETCH_HEAD
  fi
  [ "$(git -C "$CODE" rev-parse HEAD)" = "$(cat $W/cb2_l1_code.rev)" ] || { echo "STOP: code revision differs"; exit 1; }
  cd "$CODE"
  mkdir -p runs/explore/vision runs/explore/cb2/bootstrap
  cp "$W/tile_decoder_v1.pt" runs/explore/vision/
  tar --no-same-owner -C "$CODE" -xzf $W/s2b_min.tgz
  mkdir -p $W/kit && tar --no-same-owner -C $W/kit -xzf $W/cb2_kit_r2.tgz ./cb2/archive
  mv $W/kit/cb2/archive/lived-cb2-ind-0001 runs/explore/cb2/bootstrap/
  touch $W/.l1_restored
fi
cd "$CODE"
# nothing from an earlier run may still be alive (a restart once left two spawned sleeps running, which wrote into
# the new run's directory): no python, no multiprocessing children, no GPU memory in use
if pgrep -f 'cb2_l1\.py|multiprocessing|/workspace/venv/bin/python' >/dev/null; then
  echo "STOP: processes from an earlier run are still alive:"; ps -eo pid,etime,cmd | grep -E 'python|multiprocessing' | grep -v grep | cut -c1-150; exit 1
fi
if nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '$1 > 200 {b=1} END {exit !b}'; then
  echo "STOP: GPU memory is in use by something else:"; nvidia-smi; exit 1
fi
# a reused pod gets the uploaded code too, and a run never resumes across code revisions
if [ "$(git rev-parse HEAD)" != "$(cat $W/cb2_l1_code.rev)" ]; then
  git fetch -q "$W/cb2_l1_code.bundle" explore/cb2-r3 && git checkout -q --detach FETCH_HEAD
fi
[ "$(git rev-parse HEAD)" = "$(cat $W/cb2_l1_code.rev)" ] || { echo "STOP: code revision differs"; exit 1; }
if [ -f runs/explore/cb2/l1/run.json ] && ! grep -q "$(cat $W/cb2_l1_code.rev)" runs/explore/cb2/l1/run.json; then
  S=runs/explore/cb2/l1-stale-$(date -u +%Y%m%dT%H%M%S); mv runs/explore/cb2/l1 "$S"; echo "== a run from older code moved aside to $S"
fi
PINS="numpy==2.4.4 scipy==1.17.1 crafter==1.8.3 opensimplex==0.4.5.1 pillow==12.2.0 pyarrow==25.0.1 zstandard==0.25.0 jsonschema==4.26.0 pytest"
if [ ! -x $W/venv/bin/python ]; then
  if python3 -m venv --system-site-packages $W/venv 2>/dev/null; then $W/venv/bin/pip install -q $PINS
  else mkdir -p $W/venv/bin && ln -sf "$(command -v python3)" $W/venv/bin/python
       python3 -m pip install -q $PINS || python3 -m pip install -q --break-system-packages $PINS; fi
fi
export PATH=$W/venv/bin:$PATH PYTHONPATH=$CODE OPENBLAS_NUM_THREADS=1
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -m pytest -q -p no:cacheprovider tests/test_cb2_core.py tests/test_cb2_core_map.py tests/test_cb2_consolidation.py tests/test_cb2_intent.py tests/test_cb2_chooser_guards.py
python - <<'PY'
import json
m = json.load(open("runs/explore/cb2/candidates/s2b-final/cb2-ind-0001/individual.json"))
ok = m["state_hash"].startswith("cc8948e7") and m["weights_sha256"].startswith("8cffde79")
print("S2b twin source:", "OK" if ok else "MISMATCH"); raise SystemExit(0 if ok else 1)
PY
# workers from the pod's real limits (nproc shows the host's cores): cgroup CPU quota - 1, and ~1.6 GB per worker
# (one life peaks at 1.15 GB RSS); 32 workers on a 12-vCPU / 31 GB pod hung the first L1 launch
set +e +o pipefail                                   # probing optional cgroup files must not kill the run (set -e)
CG=${CG:-/sys/fs/cgroup}; N=$(nproc); Q=""
[ -r $CG/cpu.max ] && Q=$(awk '$1!="max"{print int($1/$2)}' $CG/cpu.max)
[ -z "$Q" ] && [ -r $CG/cpu/cpu.cfs_quota_us ] && Q=$(awk 'NR==FNR{q=$1;next} q>0{print int(q/$1)}' $CG/cpu/cpu.cfs_quota_us $CG/cpu/cpu.cfs_period_us)
if [ -n "$Q" ] && [ "$Q" -gt 0 ]; then [ "$Q" -lt "$N" ] && N=$Q; else N=$(( N < 12 ? N : 12 )); fi   # unknown quota: assume 12
M=""
for f in $CG/memory.max $CG/memory/memory.limit_in_bytes; do
  [ -z "$M" ] && [ -r $f ] && M=$(awk '$1!="max" && $1<1e15{print int($1/1717986918)}' $f)
done
[ -z "$M" ] && M=$(free -g | awk '/Mem:/{m=int($2/1.6); print (m < 19 ? m : 19)}')
WK=$(( N - 1 < M ? N - 1 : M )); WK=$(( WK < 1 ? 1 : WK )); WK=${WORKERS:-$WK}
set -euo pipefail
echo "cpu quota: $N  memory slots: $M"
echo "== $(date -u) running l1 with $WK workers"
python scripts/explore/cb2_l1.py run --twin runs/explore/cb2/candidates/s2b-final --bootstrap runs/explore/cb2/bootstrap \
  --out runs/explore/cb2/l1 --workers $WK --resume 2>&1 | grep --line-buffered -v '^{$\|^}$' | tee -a runs/explore/cb2/l1-run.log
python scripts/explore/cb2_l1.py report --out runs/explore/cb2/l1 --report evidence/explore/cb2/l1-report.json > /dev/null
echo "== $(date -u) gate 4 probes (learning twin's core vs the frozen core), on the learning twin's held-out lives"
python - <<'PY'
import io, json, torch, numpy as np
from pathlib import Path
for t in ("learning", "frozen"):
    h = Path(f"runs/explore/cb2/l1/twins/{t}/cb2-ind-0001"); m = json.loads((h / "individual.json").read_text())
    s = torch.load(io.BytesIO((h / "weights" / f"{m['weights_sha256']}.pt").read_bytes()), map_location="cpu", weights_only=False)
    torch.save(s["core"], f"runs/explore/cb2/l1/core-{t}.pt")
PY
for t in learning frozen; do
  python scripts/explore/cb2_sleep.py probe-life --archive runs/explore/cb2/l1/archive-learning --heldout-lives 20 \
    --core runs/explore/cb2/l1/core-$t.pt --fit-lives 120 --test-lives 20 --out evidence/explore/cb2/l1-gate4-$t.json > /dev/null || true
done
echo "== $(date -u) packing results"
tar -czf $W/cb2_l1_results.tgz runs/explore/cb2/l1 evidence/explore/cb2/l1-* runs/explore/cb2/l1-run.log
ls -la $W/cb2_l1_results.tgz && sha256sum $W/cb2_l1_results.tgz
echo "== $(date -u) l1 pod finished"
