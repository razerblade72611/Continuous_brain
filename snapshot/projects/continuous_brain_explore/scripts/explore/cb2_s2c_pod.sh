#!/usr/bin/env bash
# CB2-S2c (r6), pod side: continue from S2b, gamma 0.995, 100 iterations, one 80-world sealed test (s2c-prereg.json).
# Run ON THE POD from /workspace:
#   setsid nohup bash /workspace/cb2_s2c_pod.sh > /workspace/s2c.log 2>&1 < /dev/null &
# Needs in /workspace: tile_decoder_v1.pt (vision v1), cb2_r2_model.tar (frozen r2 core), cb2_s2b_results.tgz (the frozen
# S2b candidate), cb2_s2c_code.bundle + cb2_s2c_code.rev + cb2_s2c_SHA256SUMS.  Every input is hash-checked.
# Safe to re-run: training resumes only if selection.json is absent (it then restarts training from S2b);
# the sealed step refuses to run twice into the same directory (sealed seeds are used once).
set -euo pipefail
W=/workspace; CODE=$W/Continuous_brain
cd $W
echo "== $(date -u) s2c: verifying inputs"
sha256sum -c <<SUMS
0c46b2d1ace00c027d45e23ae01b00da8b802d10660122da49f9727a1cfe9999  tile_decoder_v1.pt
50ec80de0988aeebd2a9797483181e0a5d9890f8a820336621b0a4920c2a2cb6  cb2_r2_model.tar
4ff65462f5f7cf5479c720315d938770cd9907399d9210a8956a2dd921b63d65  cb2_s2b_results.tgz
SUMS
sha256sum -c cb2_s2c_SHA256SUMS
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "cpus: $(nproc)  mem: $(free -g | awk '/Mem:/{print $2" GB"}')"

if [ ! -f $W/.s2c_restored ]; then
  echo "== restoring code, vision v1, r2 core, frozen S2b"
  if [ ! -d "$CODE/.git" ]; then
    mkdir -p "$CODE" && git -C "$CODE" init -q && git -C "$CODE" fetch -q "$W/cb2_s2c_code.bundle" explore/cb2-r3
    git -C "$CODE" checkout -q --detach FETCH_HEAD
  fi
  [ "$(git -C "$CODE" rev-parse HEAD)" = "$(cat $W/cb2_s2c_code.rev)" ] || { echo "STOP: code revision differs from cb2_s2c_code.rev"; exit 1; }
  cd "$CODE"
  mkdir -p runs/explore/vision runs/explore/cb2/archive
  cp "$W/tile_decoder_v1.pt" runs/explore/vision/
  tar --no-same-owner -C "$CODE" -xf $W/cb2_r2_model.tar
  tar --no-same-owner -C "$CODE" -xzf $W/cb2_s2b_results.tgz runs/explore/cb2/candidates/s2b-final
  touch $W/.s2c_restored
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
echo "== $(date -u) running s2c with $WK workers"
WORKERS=$WK bash scripts/explore/cb2_r6_s2c.sh

echo "== $(date -u) packing results"
tar -czf $W/cb2_s2c_results.tgz runs/explore/cb2/s2c runs/explore/cb2/logs/s2c evidence/explore/cb2 \
    $(ls -d runs/explore/cb2/candidates/s2c-final 2>/dev/null) $(ls -d runs/explore/cb2/archive/eval-s2c-* 2>/dev/null)
ls -la $W/cb2_s2c_results.tgz && sha256sum $W/cb2_s2c_results.tgz
echo "== $(date -u) s2c pod finished"
