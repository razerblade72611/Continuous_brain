# Continuous Brain v0.2

A runnable proof-of-concept for a neural agent that **does not start from a blank prompt each cycle**. It continuously senses, maintains recurrent neural state, acts, predicts the next world state, stores surprising experiences, and learns from replay while operating.

**v0.2 adds a dependency-free live browser visualization** so you can watch the agent and its internal learning signals in real time. The neural architecture is unchanged from v0.1, so v0.1 checkpoints remain compatible.

This is an engineering prototype, not a claim of human-like consciousness.

## What v0.2 adds

- Live animated 2-D world in a browser.
- Agent trajectory and current target.
- Learned world-model **predicted next position** beside the actual position.
- Live prediction surprise/error.
- Current action and all Q-values.
- Sampled GRU recurrent-state activity.
- Reward, episode reward, goals, energy and distance.
- Replay-buffer and episodic-memory counts.
- Live learning-loss history.
- Pause/resume and 0.1x–20x speed controls.
- WSL-friendly `localhost` dashboard with no Pygame/Tk dependency.
- Optional `--open-browser` convenience flag.
- v0.1 `brain.pt` + `episodic.sqlite3` compatibility.

## Architecture

```text
world / robot / simulator
        |
        v
  observation vector
        |
        v
+---------------------+
| perception network  |
+----------+----------+
           |
           v
+---------------------+       +-------------------+
| recurrent GRU core  |<----->| working memory    |
| persistent state    |       +-------------------+
+----+-----------+----+
     |           |
     |           +--------------------+
     v                                v
+----------+                    +-------------+
| Q policy |                    | world model |
+----+-----+                    +------+------+ 
     |                                 |
     v                                 v
   action                         prediction error
     |                                 |
     +---------------+-----------------+
                     v
              replay learning
                     |
          +----------+----------+
          | episodic SQLite     |
          | memory (persistent) |
          +----------+----------+
                     |
                     v
          +---------------------+
          | browser visualizer  |
          | localhost:8765      |
          +---------------------+
```

The default environment is a small 2-D foraging world so the complete loop can run without Gym, a remote service, or robot hardware.

## Install on your Ubuntu-24.04 WSL setup

Use the project venv, not Ubuntu's system Python:

```bash
cd ~/projects/continuous_brain_project
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
```

If you are using the CPU PyTorch build:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install "numpy>=1.24"
python -m pip install -e . --no-deps
```

If you later install a known-good ROCm/CUDA-specific PyTorch build, install that Torch build first and keep using `pip install -e . --no-deps` so pip does not replace it.

## See the simulation

Start the brain with its browser dashboard:

```bash
continuous-brain \
  --state-dir state/my_brain \
  --visual
```

The terminal prints:

```text
Live visualization: http://127.0.0.1:8765
```

Open either of these in your Windows browser:

```text
http://127.0.0.1:8765
http://localhost:8765
```

With current WSL2 networking, localhost should normally forward to the service running inside WSL.

Try automatic browser launch:

```bash
continuous-brain \
  --state-dir state/my_brain \
  --visual \
  --open-browser
```

Or use the helper:

```bash
./run_visual.sh
```

### What the display means

- **Blue dot** — actual agent position.
- **Yellow crosshair/ring** — resource/goal target.
- **Pink dot** — position predicted by the learned world model after the selected action.
- **Pink dashed line** — prediction displacement from reality.
- **Faint blue trail** — recent physical trajectory.
- **Surprise** — mean-squared prediction error over the observation vector.
- **Q-values** — action-value estimates used by the learned policy.
- **Recurrent-state activity** — absolute activity of a sample of GRU hidden units.
- **Learning loss** — online world-model + reward-model + TD/Q loss.

The prediction marker is especially useful: as the world model learns, its predicted next position should generally become more consistent with actual motion, except around resets, target changes and other unexpected events.

## Visual controls

The browser dashboard has:

- **Pause / Resume** — stops cognitive ticks without killing the process.
- **Slower** — halves real-time speed, down to 0.1x.
- **Faster** — doubles real-time speed, up to 20x.
- **1x** — returns to configured `--brain-hz` timing.

The default is 10 neural ticks per second:

```bash
continuous-brain --state-dir state/my_brain --visual --brain-hz 10
```

## Run without visualization

The original headless mode still works:

```bash
continuous-brain --state-dir state/my_brain
```

Stop with `Ctrl+C`. The agent writes:

- `state/my_brain/brain.pt` — neural weights, optimizer state and counters.
- `state/my_brain/episodic.sqlite3` — durable episodic memories.
- `state/my_brain/status.json` — human-readable status.

Resume by using the same `--state-dir`.

## Fast training

```bash
continuous-brain --state-dir state/my_brain --fast --steps 50000
```

You can technically combine `--fast --visual`, but real-time visualization is most useful without `--fast` because the browser samples at a human-readable rate.

## Upgrade from v0.1 without losing the brain

The network dimensions and checkpoint format remain compatible.

Recommended safe upgrade:

```bash
cd ~/projects
mv continuous_brain_project continuous_brain_v0.1_backup
unzip /mnt/c/Users/davei/Downloads/continuous_brain_v0.2.0.zip -d ~/projects
```

Create the new venv and install your chosen Torch build, then copy the trained state:

```bash
cp -a \
  ~/projects/continuous_brain_v0.1_backup/state \
  ~/projects/continuous_brain_project/
```

Then:

```bash
cd ~/projects/continuous_brain_project
source .venv/bin/activate
continuous-brain --state-dir state/my_brain --visual
```

If you prefer, see `UPGRADE_v0.1_to_v0.2.md` for an exact sequence.

## Optional local LLM planner

The fast neural loop stays active while a local Ollama model can occasionally deliberate:

```bash
continuous-brain \
  --state-dir state/llm_brain \
  --visual \
  --ollama-model qwen3:8b \
  --planner-every 50
```

If Ollama is unavailable or times out, the neural policy keeps operating.

## What is genuinely continuous here?

1. **Persistent recurrent state** — GRU hidden state carries information across sensory ticks.
2. **No prompt/reset per tick** — observations move through one ongoing loop.
3. **World prediction** — the network predicts the next observation and reward.
4. **Surprise** — prediction error helps select events for durable episodic memory.
5. **Working memory** — recent events remain in a bounded fast-memory buffer.
6. **Episodic memory** — important events survive restarts in SQLite and are retrieved by latent similarity.
7. **Online learning** — replay updates perception, recurrent core, predictive heads and action values while the agent runs.
8. **Consolidation** — slower bursts of extra replay occur periodically.
9. **Semantic memory** — learned network parameters persist in `brain.pt`.
10. **Multiple timescales** — sensory ticks, replay, consolidation, checkpoints and optional LLM planning run at different rates.

## Connect a real simulator or robot

Implement the `Environment` interface in `continuous_brain/environment.py`:

```python
class MyWorld(Environment):
    observation_dim = 42
    action_dim = 8
    action_names = (...)

    def reset(self) -> np.ndarray:
        ...

    def step(self, action: int):
        # return observation, reward, done, info
        ...

    def describe(self) -> str:
        ...
```

Then construct `ContinuousAgent(MyWorld(), ...)`.

The current visualizer knows the demo world's `agent`, `target` and `energy` fields. When replacing the environment with a simulator such as an embodied 3-D world, the next visualization layer should use an environment-specific adapter rather than assuming 2-D coordinates.

For robotics, keep high-rate safety and servo control outside the learning agent and expose safe macro-actions to the brain.

## Tests

```bash
python -m pip install pytest
pytest -q
```

## Service mode

`continuous-brain.service.example` is still included for unattended headless operation. A browser visualizer is usually better launched interactively rather than from a login-independent service.

## Limitations / next research upgrades

v0.2 makes the continuously learning system observable; it does not solve catastrophic forgetting or human-level continual learning. Serious next upgrades include:

- sequence replay instead of independent transitions
- prioritized replay by surprise/reward
- slow/fast weights or adapters
- elastic-weight consolidation / synaptic regularization
- learned memory write/read gates
- predictive latent world model
- curiosity and homeostatic drives
- sleep/offline replay phases
- multimodal vision/audio encoders
- hierarchical skills and action arbitration
- environment-specific visual adapters
- non-learning safety supervisor beneath the learner
