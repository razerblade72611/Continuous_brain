# Exploration lane: the individual in a real world

Branch `explore/w1-world-and-organs`. Engineering only: nothing here is a scientific result, and nothing here can
touch the protected I1 lane (`LANES.md`; the lane guard test checks the frozen I1 files).

| piece | what it is | doc |
|---|---|---|
| `continuous_brain_world/` | **W1**: deterministic Crafter (pixels, survival, tech tree, day/night, monsters) plus a caretaker who sends many concurrent, changing, deadline-bound requests in English and asks about the past | `W1_WORLD.md` |
| `continuous_brain_organs/vision.py` | the **vision organ**: a learned tile decoder from pixels (0.995 / 0.9985 on held-out worlds) | `VISION_ORGAN.md` |
| `continuous_brain_organs/language.py` | the **language organ**: open-weight language models (Qwen2.5, SmolLM2, Llama-3.1 ladder) or the labelled development stand-in | `LANGUAGE_ORGAN.md` |
| `continuous_brain_individual/` | **the individual**: perception, memory, commitments, self-model, speech, skills and autobiography as hosts over the canonical store | `INDIVIDUAL_W1.md` |

## Running it on David's WSL (Ubuntu-24.04) with a real language model

Everything below happens in WSL. Open it from PowerShell with `wsl -d Ubuntu-24.04`. Use a separate checkout
from the protected I1 one, and push the branches to GitHub first (GitHub Desktop: publish
`explore/w1-world-and-organs`), or fetch them from the bundle as shown here.

```bash
cd ~/projects
git clone https://github.com/razerblade72611/Continuous_brain.git continuous_brain_explore
cd continuous_brain_explore
git fetch /mnt/c/Users/davei/cb_exchange/Continuous_brain_dev7.bundle 'refs/heads/*:refs/remotes/bundle/*' 'refs/tags/verified/*:refs/tags/verified/*'
git switch -c explore/w1-world-and-organs bundle/explore/w1-world-and-organs
python3 -m venv ~/venvs/cb_explore
source ~/venvs/cb_explore/bin/activate
pip install --upgrade pip
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
pip install numpy==2.4.4 scipy==1.17.1 crafter==1.8.3 opensimplex==0.4.5.1 jsonschema==4.26.0 transformers==5.17.0 tokenizers==0.23.2 safetensors==0.8.0 huggingface_hub==1.32.0 pytest
mkdir -p runs/explore/vision
cp /mnt/c/Users/davei/cb_exchange/tile_decoder_v1.pt runs/explore/vision/
sha256sum runs/explore/vision/tile_decoder_v1.pt
export PYTHONPATH=$PWD
python -m pytest -q -p no:cacheprovider tests/test_w1_world.py tests/test_vision_organ.py tests/test_language_organ.py tests/test_w1_individual.py
python scripts/explore/fetch_models.py --set small --models-dir ~/cb_models
python scripts/explore/w1_individual.py --seeds 71001-71005 --arms individual --language hf --model Qwen/Qwen2.5-0.5B-Instruct --local-dir ~/cb_models/Qwen--Qwen2.5-0.5B-Instruct --out runs/explore/w1_individual_qwen05
cp runs/explore/w1_individual_qwen05/report.json /mnt/c/Users/davei/cb_exchange/w1_individual_qwen05.json
cp ~/cb_models/models.lock.json /mnt/c/Users/davei/cb_exchange/models.lock.json
```

What to check:

- **The weights hash** must be `0c46b2d1ace00c027d45e23ae01b00da8b802d10660122da49f9727a1cfe9999`.
- **The tests** must all pass.
- **The run** takes a few minutes per life on CPU: the model reads each caretaker message, about 30 per life.
- **Afterwards**, send `w1_individual_qwen05.json` and `models.lock.json` back. The lock file pins the exact
  model revision and file hashes that were downloaded.

The pip pins matter: Crafter's world generation depends on numpy, so the same seeds give the same worlds only
with the same versions.
