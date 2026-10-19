# Continuous Brain v0.1

A runnable proof-of-concept for a neural agent that **does not start from a blank prompt each cycle**. It continuously senses, updates persistent recurrent state, chooses actions, predicts the next world state, records surprising experiences, and learns from replay while it is operating.

This is an engineering prototype, not a claim of human-like consciousness.

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
          +---------------------+
```

The default demo is a small 2-D foraging world so the complete loop can run without Gym, a simulator, network access, or hardware.

## Install

Python 3.10+ is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

If you already have a ROCm/CUDA-specific PyTorch installation, install that build first and then use `pip install -e . --no-deps` if you do not want pip to replace it.

## Run forever

```bash
continuous-brain --state-dir state/my_brain
```

Stop with `Ctrl+C`. The agent writes:

- `state/my_brain/brain.pt` — neural weights, optimizer state, counters
- `state/my_brain/episodic.sqlite3` — long-lived episodic memories
- `state/my_brain/status.json` — human-readable run status

Run at another cognitive tick rate:

```bash
continuous-brain --brain-hz 20
```

Run as fast as the machine permits for training:

```bash
continuous-brain --fast --steps 50000
```

Resume simply by starting it again with the same `--state-dir`.

## Optional local LLM planner

The fast neural loop remains in control. An Ollama model can be asked at a much slower interval to act as a deliberative planner:

```bash
continuous-brain \
  --state-dir state/llm_brain \
  --ollama-model qwen3:8b \
  --planner-every 50
```

If Ollama is down or times out, the continuous neural policy keeps operating.

## What is genuinely continuous here?

1. **Persistent recurrent state** — the GRU hidden state carries information from one sensory tick into the next.
2. **No prompt/reset per tick** — observations flow through one ongoing cognitive loop.
3. **World prediction** — the network predicts the next observation and reward.
4. **Surprise** — prediction error decides which moments deserve durable episodic memory.
5. **Working memory** — recent events live in a bounded fast-memory buffer.
6. **Episodic memory** — important events survive process restarts in SQLite and are retrievable by latent similarity.
7. **Online learning** — replay batches update the neural network while the agent is running.
8. **Consolidation** — slower bursts of extra replay occur periodically.
9. **Semantic memory** — learned neural weights persist in `brain.pt`.
10. **Multiple timescales** — neural ticks, ordinary replay, consolidation, checkpoints, and optional LLM planning run at different rates.

## Connect it to a real simulator or robot

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

For vision, do not feed raw megapixel images directly into this tiny GRU. Put a vision encoder in front of it and pass its compact embedding as the observation vector. For robotics, keep high-rate safety and servo control outside the learning agent; expose safe macro-actions to the brain.

## Important limitations

This v0.1 solves the systems problem of continuous operation, memory, persistence and online replay. It does **not** solve continual learning in the research sense. Catastrophic forgetting can still happen. The next serious upgrades would be:

- sequence replay rather than isolated latent transitions
- prioritized replay by surprise and reward
- separate slow/fast weights or adapters
- elastic-weight consolidation / synaptic regularization
- learned memory write/read gates
- a larger predictive latent world model
- intrinsic curiosity and homeostatic drives
- sleep/offline replay phases
- multimodal sensor encoders
- hierarchical skills and action arbitration
- safety supervisor separated from the learner

## Tests

```bash
pip install pytest
pytest -q
```

## Keep it alive across logins/reboots (Linux systemd user service)

An example `continuous-brain.service.example` is included. After placing the project at `~/continuous_brain_project` and creating its `.venv`:

```bash
mkdir -p ~/.config/systemd/user
cp continuous-brain.service.example ~/.config/systemd/user/continuous-brain.service
systemctl --user daemon-reload
systemctl --user enable --now continuous-brain.service
systemctl --user status continuous-brain.service
```

Follow its live output with:

```bash
journalctl --user -u continuous-brain.service -f
```

Stop it cleanly with:

```bash
systemctl --user stop continuous-brain.service
```

The process receives SIGTERM, checkpoints the network, writes status, and closes the episodic-memory database before exit.
