#!/usr/bin/env bash
# On the GPU pod, in /workspace, after receiving the kit:
#   mkdir -p /workspace/kit && tar -C /workspace/kit -xzf cb2_kit_r1.tgz && bash /workspace/kit/cb2_pod_run.sh r1
# Sets up the code and a pinned environment, runs the whole round (scripts/explore/cb2_s1.sh), and packs what it
# learned and measured into /workspace/cb2_results_<round>.tgz.  Safe to re-run after an interruption: finished
# steps are skipped.  The canonical individual is never changed here: the round installs into a candidate copy.
set -euo pipefail
ROUND=${1:?round label, e.g. r1}
KIT=/workspace/kit
CODE=/workspace/Continuous_brain
if [ ! -d "$CODE/.git" ]; then
  git clone -q "$KIT/code.bundle" "$CODE" -b "$(cat "$KIT/code.branch")"
fi
cd "$CODE"
[ "$(git rev-parse HEAD)" = "$(cat "$KIT/code.rev")" ] || { echo "code revision differs from the kit's"; exit 1; }
mkdir -p runs/explore/vision runs/explore/cb2/archive
[ -f runs/explore/vision/tile_decoder_v1.pt ] || cp "$KIT/tile_decoder_v1.pt" runs/explore/vision/
[ -d runs/explore/cb2/individuals ] || cp -r "$KIT/cb2/individuals" runs/explore/cb2/
for d in "$KIT"/cb2/archive/lived-*; do [ -d "runs/explore/cb2/archive/$(basename "$d")" ] || cp -r "$d" runs/explore/cb2/archive/; done
echo "0c46b2d1ace00c027d45e23ae01b00da8b802d10660122da49f9727a1cfe9999  runs/explore/vision/tile_decoder_v1.pt" | sha256sum -c
PINS="numpy==2.4.4 crafter==1.8.3 opensimplex==0.4.5.1 pillow==12.2.0 pyarrow==25.0.1 zstandard==0.25.0 jsonschema==4.26.0 pytest"
if [ ! -x /workspace/venv/bin/python ]; then
  if python3 -m venv --system-site-packages /workspace/venv 2>/dev/null; then   # keeps the pod's CUDA PyTorch
    /workspace/venv/bin/pip install -q $PINS
  else                                               # no venv module: the pod is disposable, use its Python
    mkdir -p /workspace/venv/bin && ln -sf "$(command -v python3)" /workspace/venv/bin/python
    python3 -m pip install -q $PINS || python3 -m pip install -q --break-system-packages $PINS
  fi
fi
export PATH=/workspace/venv/bin:$PATH
export PYTHONPATH=$CODE OPENBLAS_NUM_THREADS=1
python -c "import torch; assert torch.cuda.is_available(), 'no GPU visible'; print(torch.__version__, torch.cuda.get_device_name(0))"
python -m pytest -q -p no:cacheprovider tests/test_cb2_archive.py tests/test_cb2_core.py tests/test_cb2_consolidation.py
python -c "from continuous_brain_cb2.archive import Archive; print(Archive('runs/explore/cb2/archive').verify())"
bash scripts/explore/cb2_s1.sh "$ROUND" "${STEPS:-1000000}" "${SIZE:-S}" "${CORE_UPDATES:-30000}" "${POLICY_UPDATES:-15000}"
tar -C "$CODE" -czf "/workspace/cb2_results_$ROUND.tgz" "runs/explore/cb2/sleep/$ROUND" "runs/explore/cb2/candidates/$ROUND" \
    "runs/explore/cb2/logs/$ROUND" evidence/explore/cb2
echo "results: /workspace/cb2_results_$ROUND.tgz ($(du -h /workspace/cb2_results_$ROUND.tgz | cut -f1))"
