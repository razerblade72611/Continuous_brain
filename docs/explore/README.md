# Exploration lane: the individual in a real world

Branch `explore/w1-world-and-organs`. Engineering only: nothing here is a scientific result, and nothing here can
touch the protected I1 lane (`LANES.md`; the lane guard test checks the frozen I1 files).

| piece | what it is | doc |
|---|---|---|
| `continuous_brain_world/` | **W1**: deterministic Crafter (pixels, survival, tech tree, day/night, monsters) plus a caretaker who sends many concurrent, changing, deadline-bound requests in English and asks about the past | `W1_WORLD.md` |
| `continuous_brain_organs/vision.py` | the **vision organ**: a learned tile decoder from pixels (0.995 / 0.9985 on held-out worlds) | `VISION_ORGAN.md` |
| `continuous_brain_organs/language.py` | the **language organ**: open-weight language models (Qwen2.5, SmolLM2, Llama-3.1 ladder) or the labelled development stand-in | `LANGUAGE_ORGAN.md` |
| `continuous_brain_individual/` | **the individual**: perception, memory, commitments, self-model, speech, skills and autobiography as hosts over the canonical store | `INDIVIDUAL_W1.md` |
| `continuous_brain_individual/learned.py`, `scripts/explore/w1_lifelong.py` | **one individual across lives**: learned foresight and choice, trained only on its own lives | `LIFELONG_W1.md` |
| `continuous_brain_individual/viewer.py`, `scripts/explore/w1_viewer.py` | **the live viewer**: watch it live, learn and continue across lives in the browser (below) | this file |
| `continuous_brain_cb2/`, `scripts/explore/cb2_*.py` | **CB2-S1**: the scale-first kernel: experience archive, DreamerV3 predictive core, imitation skill, consolidation | `CB2_S1.md` |

## Setting up on David's WSL (Ubuntu-24.04)

Everything below happens in WSL. Open it from PowerShell with `wsl -d Ubuntu-24.04`. The exploration lane gets its
own checkout and its own Python environment, separate from the protected I1 checkout. The clone comes from a
self-contained bundle in `cb_exchange`, so it needs no GitHub access.

```bash
mkdir -p ~/projects && cd ~/projects
git clone -b explore/w1-world-and-organs /mnt/c/Users/davei/cb_exchange/Continuous_brain_explore.bundle continuous_brain_explore
cd ~/projects/continuous_brain_explore
python3 -m venv ~/venvs/cb_explore
source ~/venvs/cb_explore/bin/activate
pip install --upgrade pip
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
pip install numpy==2.4.4 scipy==1.17.1 crafter==1.8.3 opensimplex==0.4.5.1 pillow==12.2.0 jsonschema==4.26.0 transformers==5.17.0 tokenizers==0.23.2 safetensors==0.8.0 huggingface_hub==1.32.0 pytest
mkdir -p runs/explore/vision
cp /mnt/c/Users/davei/cb_exchange/tile_decoder_v1.pt runs/explore/vision/
sha256sum runs/explore/vision/tile_decoder_v1.pt
export PYTHONPATH=$PWD
python -m pytest -q -p no:cacheprovider tests/test_w1_world.py tests/test_vision_organ.py tests/test_language_organ.py tests/test_w1_individual.py tests/test_w1_learned.py tests/test_w1_viewer.py
```

- **The weights hash** must be `0c46b2d1ace00c027d45e23ae01b00da8b802d10660122da49f9727a1cfe9999`.
- **The tests** must all pass.
- **The pip pins matter**: Crafter's world generation depends on numpy, so the same seeds give the same worlds
  only with the same versions.

To update an existing checkout from a newer bundle instead of cloning again:

```bash
cd ~/projects/continuous_brain_explore
git pull /mnt/c/Users/davei/cb_exchange/Continuous_brain_explore.bundle explore/w1-world-and-organs
```

## Watching it live

```bash
cd ~/projects/continuous_brain_explore
source ~/venvs/cb_explore/bin/activate
export PYTHONPATH=$PWD
python scripts/explore/w1_viewer.py --seed 71001
```

The CB2 individual (`CB2_S1.md`) lives its own next lives in the same viewer, every tick archived:
`python scripts/explore/w1_viewer.py --individual cb2-ind-0001`.

Then open http://localhost:8765 in a Windows browser (WSL forwards localhost). If the page doesn't load, stop the
server with Ctrl+C and start it again with `--host 0.0.0.0` added.

**One individual, many lives.** By default the individual learns from its own lives (`LIFELONG_W1.md`). When it
dies it has a last moment in which it closes the life and learns from it. Then the whole individual is saved to disk
automatically. With "Next life when it dies" checked, the same individual wakes in the next world. Its map,
requests and plans belong to the old world and start over; what it learned, what it believes about itself and
its diary carry on. Ctrl+C also saves it. To pick up where you left off, the next day or after a restart:

```bash
python scripts/explore/w1_viewer.py --resume latest
```

`--skills routine` gives the earlier individual with the fixed routine library and one life.

What the page shows, top to bottom:

- **What it sees, what it remembers, what is really there.** Its actual 64 by 64 input with what its vision organ
  perceives in each tile (mistakes framed in red), its remembered map (built only from what it has seen), and the
  true world, which is for you only and never reaches the individual.
- **What it is doing now**: the action, the routine or exploration it serves, the commitment it is working on,
  what it wants, and what it just said.
- **How its faculties are wired**: the real channel wiring between its faculties in the canonical store; each box
  fills when that faculty's state changed this tick, with the milliseconds it took.
- **Its activity over time**: every number each faculty writes to the store, one row each, one column per tick,
  plus the time each faculty spends; hits (red), messages heard (pink) and first achievements (yellow) are
  marked underneath.
- **What it foresees, how it values its choices, and its lives**: its learned predictions of dying, running
  out of water or food, getting hurt or exhausted in the next 100 steps; the value it puts on each option now (the
  one it is following is marked); and every life so far, how long it lived and how much it did, with the average
  of the last ten lives, which is where learning shows.
- **Body, commitments, conversation, self-model and diary**: vitals over time, every request with its deadline
  and both its own and the caretaker's verdict, the conversation with what it understood from each message, its
  believed success rates, and the days it has narrated.

Controls: Play and Pause (Space), Step (the period key), speed, "Next life when it dies", starting a new
individual (a newborn: seed, ablations, learned or fixed skills), erasing one faculty's state in the store
(effective from the next tick; erasing Skills or Foresight also forgets what they learned), a bookmark to return
to, and Save to disk.

Saves go to `runs/explore/viewer/lives/<individual>/checkpoints/<snapshot>`, with one line per life in
`runs/explore/viewer/lives/<individual>/lives.jsonl`. They are never overwritten, and the runtime
snapshot is checksummed. Resume one with:

```bash
python scripts/explore/w1_viewer.py --resume runs/explore/viewer/lives/<individual>/checkpoints/<snapshot>
```

Watching does not change the life: `tests/test_w1_viewer.py` checks that a watched life is identical, action by
action and in its final state hash, to an unwatched one.

## With a real language model

```bash
cd ~/projects/continuous_brain_explore
source ~/venvs/cb_explore/bin/activate
export PYTHONPATH=$PWD
python scripts/explore/fetch_models.py --set small --models-dir ~/cb_models
python scripts/explore/w1_viewer.py --seed 71001 --language hf --model Qwen/Qwen2.5-0.5B-Instruct --local-dir ~/cb_models/Qwen--Qwen2.5-0.5B-Instruct
python scripts/explore/w1_individual.py --seeds 71001-71005 --arms individual --language hf --model Qwen/Qwen2.5-0.5B-Instruct --local-dir ~/cb_models/Qwen--Qwen2.5-0.5B-Instruct --out runs/explore/w1_individual_qwen05
cp runs/explore/w1_individual_qwen05/report.json /mnt/c/Users/davei/cb_exchange/w1_individual_qwen05.json
cp ~/cb_models/models.lock.json /mnt/c/Users/davei/cb_exchange/models.lock.json
```

- **The viewer** shows the model's name at the bottom of the page, and each caretaker message with what the model
  understood from it.
- **The experiment** takes a few minutes per life on CPU: the model reads each caretaker message, about 30 per life.
- **Afterwards**, send `w1_individual_qwen05.json` and `models.lock.json` back. The lock file pins the exact
  model revision and file hashes that were downloaded.

## Training it for many lives, headless

Your machine is much faster than the cloud workspace. This lets one individual live 2000 lives on training worlds,
saving it every 25 lives. It can be stopped with Ctrl+C and continued with `--resume <snapshot>`.

```bash
cd ~/projects/continuous_brain_explore
source ~/venvs/cb_explore/bin/activate
export PYTHONPATH=$PWD
python scripts/explore/w1_lifelong.py train --lives 2000 --out runs/explore/lifelong/home1
```

Then test it frozen on the 20 test worlds (it never trained on them), and watch it:

```bash
ls runs/explore/lifelong/home1/individual/checkpoints/
python scripts/explore/w1_lifelong.py evaluate --snapshot runs/explore/lifelong/home1/individual/checkpoints/<snapshot>
python scripts/explore/w1_viewer.py --adopt runs/explore/lifelong/home1/individual/checkpoints/<snapshot>
```

Each life is one line in `runs/explore/lifelong/home1/lives.jsonl`. Send that file and the `.eval.json` files
back to compare with `LIFELONG_W1.md`.
