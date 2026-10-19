#!/usr/bin/env bash
# CB2 r3-H arm B (learned selector -> scripted executor organ), pod side.  Run ON THE POD from /workspace:
#   setsid nohup bash /workspace/cb2_r3h_pod.sh > /workspace/r3h.log 2>&1 < /dev/null &
# Restores code (explore/cb2-r3 bundle), the individual (r2 kit) and the frozen r2 core + r2 P-A policy
# (cb2_r2_model.tar, hash-checked against r2's own .sha256 files); runs cb2_r3_hier.sh; packs the results.
# r3-H regenerates practice run 4 (same seeds as r3-T) with organ-command labels.  Safe to re-run.
set -euo pipefail
W=/workspace; KIT=$W/kit; CODE=$W/Continuous_brain
cd $W
echo "== $(date -u) r3h: verifying inputs"
sha256sum -c <<SUMS
2d3ec50ce05e635a46fc2d658a2395aa65de929088c17879090e751f15a1a1ec  cb2_kit_r2.tgz
50ec80de0988aeebd2a9797483181e0a5d9890f8a820336621b0a4920c2a2cb6  cb2_r2_model.tar
SUMS
sha256sum -c cb2_r3h_SHA256SUMS
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "cpus: $(nproc)  mem: $(free -g | awk '/Mem:/{print $2" GB"}')"

if [ ! -f $W/.r3h_restored ]; then
  echo "== restoring code, individual, r2 core and r2 P-A policy"
  mkdir -p $KIT && tar --no-same-owner -C $KIT -xzf cb2_kit_r2.tgz ./tile_decoder_v1.pt ./cb2/individuals
  if [ ! -d "$CODE/.git" ]; then
    mkdir -p "$CODE" && git -C "$CODE" init -q && git -C "$CODE" fetch -q "$W/cb2_r3h_code.bundle" explore/cb2-r3
    git -C "$CODE" checkout -q --detach FETCH_HEAD
  fi
  [ "$(git -C "$CODE" rev-parse HEAD)" = "$(cat $W/cb2_r3h_code.rev)" ] || { echo "STOP: code revision differs from cb2_r3h_code.rev"; exit 1; }
  cd "$CODE"
  mkdir -p runs/explore/vision runs/explore/cb2/archive
  cp "$KIT/tile_decoder_v1.pt" runs/explore/vision/
  echo "0c46b2d1ace00c027d45e23ae01b00da8b802d10660122da49f9727a1cfe9999  runs/explore/vision/tile_decoder_v1.pt" | sha256sum -c
  cp -r "$KIT/cb2/individuals" runs/explore/cb2/
  tar --no-same-owner -C "$CODE" -xf $W/cb2_r2_model.tar
  touch $W/.r3h_restored
fi
cd "$CODE"
for f in runs/explore/cb2/sleep/r2/core/core.pt runs/explore/cb2/sleep/r2/policy-P-A/policy.pt; do
  [ "$(cut -d' ' -f1 $f.sha256)" = "$(sha256sum $f | cut -d' ' -f1)" ] && echo "$f: OK" || { echo "STOP: $f does not match its hash"; exit 1; }
done
grep -q '"state_hash": "b7beb85d3da691780930d0b5c5ea2784cdb53b147950c481d5014f9442eade09"' \
  runs/explore/cb2/individuals/cb2-ind-0001/individual.json && echo "individual cb2-ind-0001: OK" || { echo "STOP: not the recorded individual"; exit 1; }

PINS="numpy==2.4.4 scipy==1.17.1 crafter==1.8.3 opensimplex==0.4.5.1 pillow==12.2.0 pyarrow==25.0.1 zstandard==0.25.0 jsonschema==4.26.0 pytest"
if [ ! -x $W/venv/bin/python ]; then
  if python3 -m venv --system-site-packages $W/venv 2>/dev/null; then $W/venv/bin/pip install -q $PINS
  else mkdir -p $W/venv/bin && ln -sf "$(command -v python3)" $W/venv/bin/python
       python3 -m pip install -q $PINS || python3 -m pip install -q --break-system-packages $PINS; fi
fi
export PATH=$W/venv/bin:$PATH PYTHONPATH=$CODE OPENBLAS_NUM_THREADS=1
python -c "import torch; assert torch.cuda.is_available(), 'no GPU visible'; print(torch.__version__, torch.cuda.get_device_name(0))"
python -m pytest -q -p no:cacheprovider tests/test_cb2_archive.py tests/test_cb2_core.py tests/test_cb2_consolidation.py \
  tests/test_cb2_intent.py
python -c "from continuous_brain_cb2.organ import COMMANDS; assert 'routine' not in COMMANDS; print('organ commands:', COMMANDS)"

echo "== $(date -u) running r3h"
WORKERS=${WORKERS:-24} bash scripts/explore/cb2_r3_hier.sh

echo "== $(date -u) packing results"
tar -czf $W/cb2_r3h_results.tgz runs/explore/cb2/sleep/r3h runs/explore/cb2/logs/r3h evidence/explore/cb2 \
    runs/explore/cb2/intent_sidecar runs/explore/cb2/candidates/r3h-B $(ls -d runs/explore/cb2/archive/eval-r3h-* 2>/dev/null)
ls -la $W/cb2_r3h_results.tgz && sha256sum $W/cb2_r3h_results.tgz
echo "== $(date -u) r3h pod finished"
