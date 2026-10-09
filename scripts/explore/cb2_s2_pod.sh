#!/usr/bin/env bash
# CB2-S2 (r4), pod side: PPO + KL to frozen r3h-B, development selection, one sealed test (s2-prereg.json).
# Run ON THE POD from /workspace:
#   setsid nohup bash /workspace/cb2_s2_pod.sh > /workspace/s2.log 2>&1 < /dev/null &
# Needs in /workspace: cb2_kit_r2.tgz (vision v1), cb2_r2_model.tar (frozen r2 core), cb2_r3h_results.tgz (the frozen
# r3h-B candidate), cb2_s2_code.bundle + cb2_s2_code.rev + cb2_s2_SHA256SUMS.  Every input is hash-checked.
# Safe to re-run: training resumes only if selection.json is absent (it then restarts training from r3h-B);
# the sealed step refuses to run twice into the same directory (sealed seeds are used once).
set -euo pipefail
W=/workspace; KIT=$W/kit; CODE=$W/Continuous_brain
cd $W
echo "== $(date -u) s2: verifying inputs"
sha256sum -c <<SUMS
2d3ec50ce05e635a46fc2d658a2395aa65de929088c17879090e751f15a1a1ec  cb2_kit_r2.tgz
50ec80de0988aeebd2a9797483181e0a5d9890f8a820336621b0a4920c2a2cb6  cb2_r2_model.tar
71f215f61b04af7ec66a97615364f30481f42803f3f9812c5e0c5537bf8438ef  cb2_r3h_results.tgz
SUMS
sha256sum -c cb2_s2_SHA256SUMS
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "cpus: $(nproc)  mem: $(free -g | awk '/Mem:/{print $2" GB"}')"

if [ ! -f $W/.s2_restored ]; then
  echo "== restoring code, vision v1, r2 core, frozen r3h-B"
  mkdir -p $KIT && tar --no-same-owner -C $KIT -xzf cb2_kit_r2.tgz ./tile_decoder_v1.pt
  if [ ! -d "$CODE/.git" ]; then
    mkdir -p "$CODE" && git -C "$CODE" init -q && git -C "$CODE" fetch -q "$W/cb2_s2_code.bundle" explore/cb2-r3
    git -C "$CODE" checkout -q --detach FETCH_HEAD
  fi
  [ "$(git -C "$CODE" rev-parse HEAD)" = "$(cat $W/cb2_s2_code.rev)" ] || { echo "STOP: code revision differs from cb2_s2_code.rev"; exit 1; }
  cd "$CODE"
  mkdir -p runs/explore/vision runs/explore/cb2/archive
  cp "$KIT/tile_decoder_v1.pt" runs/explore/vision/
  echo "0c46b2d1ace00c027d45e23ae01b00da8b802d10660122da49f9727a1cfe9999  runs/explore/vision/tile_decoder_v1.pt" | sha256sum -c
  tar --no-same-owner -C "$CODE" -xf $W/cb2_r2_model.tar
  tar --no-same-owner -C "$CODE" -xzf $W/cb2_r3h_results.tgz runs/explore/cb2/candidates/r3h-B
  touch $W/.s2_restored
fi
cd "$CODE"
f=runs/explore/cb2/sleep/r2/core/core.pt
[ "$(cut -d' ' -f1 $f.sha256)" = "$(sha256sum $f | cut -d' ' -f1)" ] && echo "$f: OK" || { echo "STOP: $f does not match its hash"; exit 1; }

PINS="numpy==2.4.4 scipy==1.17.1 crafter==1.8.3 opensimplex==0.4.5.1 pillow==12.2.0 pyarrow==25.0.1 zstandard==0.25.0 jsonschema==4.26.0 pytest"
if [ ! -x $W/venv/bin/python ]; then
  if python3 -m venv --system-site-packages $W/venv 2>/dev/null; then $W/venv/bin/pip install -q $PINS
  else mkdir -p $W/venv/bin && ln -sf "$(command -v python3)" $W/venv/bin/python
       python3 -m pip install -q $PINS || python3 -m pip install -q --break-system-packages $PINS; fi
fi
export PATH=$W/venv/bin:$PATH PYTHONPATH=$CODE OPENBLAS_NUM_THREADS=1
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -m pytest -q -p no:cacheprovider tests/test_cb2_core.py tests/test_cb2_consolidation.py tests/test_cb2_intent.py \
  tests/test_cb2_organ_swap.py
python -c "from continuous_brain_cb2.organ import COMMANDS; assert 'routine' not in COMMANDS; print('organ commands:', COMMANDS)"

N=$(nproc); WK=${WORKERS:-$(( N > 40 ? 40 : N - 2 ))}
echo "== $(date -u) running s2 with $WK workers"
WORKERS=$WK bash scripts/explore/cb2_r4_s2.sh

echo "== $(date -u) packing results"
tar -czf $W/cb2_s2_results.tgz runs/explore/cb2/s2 runs/explore/cb2/logs/s2 evidence/explore/cb2 \
    $(ls -d runs/explore/cb2/candidates/s2-final 2>/dev/null) $(ls -d runs/explore/cb2/archive/eval-s2-* 2>/dev/null)
ls -la $W/cb2_s2_results.tgz && sha256sum $W/cb2_s2_results.tgz
echo "== $(date -u) s2 pod finished"
