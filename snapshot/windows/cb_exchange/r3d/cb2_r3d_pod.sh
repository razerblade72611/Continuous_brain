#!/usr/bin/env bash
# CB2 r3-D, pod side.  Run ON THE POD from /workspace:
#   setsid nohup bash /workspace/cb2_r3d_pod.sh > /workspace/r3d.log 2>&1 < /dev/null &
# Restores code (explore/cb2-r3 bundle), data (r2 kit), the frozen r2 core and r2 P-A policy; regenerates practice
# run 2 exactly as in r2 and checks it file by file; runs cb2_r3_dagger.sh; packs the results.  Safe to re-run.
set -euo pipefail
W=/workspace; KIT=$W/kit; CODE=$W/Continuous_brain
cd $W
echo "== $(date -u) r3d: verifying inputs"
sha256sum -c <<SUMS
2d3ec50ce05e635a46fc2d658a2395aa65de929088c17879090e751f15a1a1ec  cb2_kit_r2.tgz
d6874bf6d1da22d100f1dae12446317f2bca11c9ea232a860b7c4998389f2ef8  cb2_partial_r2.tgz
390722b2a0a15249472f71430a9598c90c0840f3d58134fdaf11e956e63cf2aa  practice_r2_manifests.tgz
SUMS
sha256sum -c cb2_r3d_SHA256SUMS
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "cpu quota: $(cat /sys/fs/cgroup/cpu.max 2>/dev/null)  mem limit: $(cat /sys/fs/cgroup/memory.max 2>/dev/null)"

if [ ! -f $W/.r3d_restored ]; then
  echo "== restoring code, data, r2 core and r2 P-A policy"
  mkdir -p $KIT && tar --no-same-owner -C $KIT -xzf cb2_kit_r2.tgz
  if [ ! -d "$CODE/.git" ]; then
    mkdir -p "$CODE" && git -C "$CODE" init -q && git -C "$CODE" fetch -q "$W/cb2_r3d_code.bundle" explore/cb2-r3
    git -C "$CODE" checkout -q --detach FETCH_HEAD
  fi
  [ "$(git -C "$CODE" rev-parse HEAD)" = "$(cat $W/cb2_r3d_code.rev)" ] || { echo "STOP: code revision differs from cb2_r3d_code.rev"; exit 1; }
  cd "$CODE"
  mkdir -p runs/explore/vision runs/explore/cb2/archive
  cp "$KIT/tile_decoder_v1.pt" runs/explore/vision/
  echo "0c46b2d1ace00c027d45e23ae01b00da8b802d10660122da49f9727a1cfe9999  runs/explore/vision/tile_decoder_v1.pt" | sha256sum -c
  cp -r "$KIT/cb2/individuals" runs/explore/cb2/
  for d in "$KIT"/cb2/archive/lived-*; do cp -r "$d" runs/explore/cb2/archive/; done
  tar --no-same-owner -C $W -xzf $W/cb2_partial_r2.tgz --wildcards \
      'Continuous_brain/runs/explore/cb2/sleep/r2/core/core.pt*' 'Continuous_brain/runs/explore/cb2/sleep/r2/core/run.json' \
      'Continuous_brain/runs/explore/cb2/sleep/r2/policy-P-A/*'
  touch $W/.r3d_restored
fi
cd "$CODE"
for f in runs/explore/cb2/sleep/r2/core/core.pt runs/explore/cb2/sleep/r2/policy-P-A/policy.pt; do
  [ "$(cut -d' ' -f1 $f.sha256)" = "$(sha256sum $f | cut -d' ' -f1)" ] && echo "$f: OK" || { echo "STOP: $f does not match its hash"; exit 1; }
done

PINS="numpy==2.4.4 scipy==1.17.1 crafter==1.8.3 opensimplex==0.4.5.1 pillow==12.2.0 pyarrow==25.0.1 zstandard==0.25.0 jsonschema==4.26.0 pytest"
if [ ! -x $W/venv/bin/python ]; then
  if python3 -m venv --system-site-packages $W/venv 2>/dev/null; then $W/venv/bin/pip install -q $PINS
  else mkdir -p $W/venv/bin && ln -sf "$(command -v python3)" $W/venv/bin/python
       python3 -m pip install -q $PINS || python3 -m pip install -q --break-system-packages $PINS; fi
fi
export PATH=$W/venv/bin:$PATH PYTHONPATH=$CODE OPENBLAS_NUM_THREADS=1
python -c "import torch; assert torch.cuda.is_available(), 'no GPU visible'; print(torch.__version__, torch.cuda.get_device_name(0))"
python -m pytest -q -p no:cacheprovider tests/test_cb2_archive.py tests/test_cb2_core.py tests/test_cb2_consolidation.py
python -c "from continuous_brain_cb2.archive import Archive; print(Archive('runs/explore/cb2/archive').verify())"

if [ ! -f $W/.r3d_practice_ok ]; then
  echo "== $(date -u) regenerating practice run 2 (the base data the r2 policy was trained on)"
  ls -d runs/explore/cb2/archive/practice-r002-w* >/dev/null 2>&1 || \
    python scripts/explore/cb2_collect.py practice --workers 24 --steps 1000000 --run 2
  rm -rf $W/practice_r2_orig && mkdir -p $W/practice_r2_orig && tar -C $W/practice_r2_orig -xzf $W/practice_r2_manifests.tgz
  python - <<'PY'
import json, pathlib, sys
orig, new = pathlib.Path("/workspace/practice_r2_orig"), pathlib.Path("/workspace/Continuous_brain")
bad = [str(m.relative_to(orig)) for m in sorted(orig.rglob("*.json"))
       if not (new / m.relative_to(orig)).exists()
       or {k: v.get("sha256") for k, v in json.loads(m.read_text()).get("files", {}).items()}
       != {k: v.get("sha256") for k, v in json.loads((new / m.relative_to(orig)).read_text()).get("files", {}).items()}]
print(json.dumps({"practice_manifests_checked": len(list(orig.rglob("*.json"))), "differ_or_missing": bad}))
sys.exit(1 if bad else 0)
PY
  touch $W/.r3d_practice_ok
fi

echo "== $(date -u) running r3d"
WORKERS=24 bash scripts/explore/cb2_r3_dagger.sh

echo "== $(date -u) packing results"
tar -czf $W/cb2_r3d_results.tgz runs/explore/cb2/sleep/r3d runs/explore/cb2/logs/r3d evidence/explore/cb2 \
    runs/explore/cb2/dagger_archive runs/explore/cb2/candidates/r3d-dagger runs/explore/cb2/candidates/r3d-control \
    $(ls -d runs/explore/cb2/archive/eval-r3d-* 2>/dev/null)
ls -la $W/cb2_r3d_results.tgz && sha256sum $W/cb2_r3d_results.tgz
echo "== $(date -u) r3d pod finished"
