# Continuous Brain v0.4.1

Continuous Brain is an experimental persistent neural-agent architecture that combines recurrent lifetime state, online learning, predictive modeling, episodic memory, intrinsic motivation, optional frozen foundation perception/language models, and now a 3-D embodiment boundary.

v0.4.1 keeps the v0.3 brain architecture recognizable while moving the environment from a 2-D demonstration toward RGB-D embodied navigation.

## Architecture

```text
                         3-D WORLD
                      AI2-THOR v0.4.1
                           │
             ┌─────────────┼─────────────┐
             │             │             │
            RGB          depth       pose/action
             │             │             │
             ▼             │             │
     frozen visual model   │             │
      (optional DINOv3)    │             │
             │             │             │
             ▼             ▼             ▼
       visual adapter   embodiment sensor adapter
             │             │
             └──────┬──────┘
                    ▼
             persistent GRU
                    │
       ┌────────────┼────────────┐
       ▼            ▼            ▼
     policy      world model    memory
       │
       ▼
  5 learned intents
 idle / advance / turn L / turn R / interact
       │
       ▼
 embodiment executor + safety boundary
       │
       ▼
 simulator primitive actions
```

## Why keep six structured inputs and five intents?

Changing from the 2-D demo to a 3-D body does not require discarding everything already learned. The embodiment adapter compresses current task/sensor information into the same six structured slots used by the original brain while RGB is handled by the separate foundation-vision pathway.

For AI2-THOR the six values are:

1. goal-forward component
2. goal-right component
3. center depth clearance
4. left/right depth imbalance
5. previous action success
6. previous extrinsic reward

The two goal slots can be hidden completely (`sensor-only`) or randomly dropped during training. This lets us use strong simulator supervision early and progressively remove it.

## v0.4.1 curriculum correction

v0.4.1 makes the curriculum constrain the policy action space, not merely the reward label. The 5-output head is preserved for checkpoint compatibility, but invalid actions are masked during epsilon exploration, greedy Q selection, TD targets and optional slow planning.

| Level | Allowed intents |
|---:|---|
| 0 | idle, turn_left, turn_right |
| 1–5 | idle, advance, turn_left, turn_right |
| 6–9 | all five intents including interact |

L0 also anchors the agent position, uses stronger angular shaping, and terminates stagnant episodes.

Default graduation readiness uses a rolling 100-episode window: at least 95% success, median successful completion no more than 6 steps, and timeout rate no more than 2%. Statistics persist in the v0.4.1 checkpoint.

## Curriculum

| Level | Task |
|---:|---|
| 0 | orient toward a goal direction |
| 1 | navigate to a reachable point |
| 2 | approach an initially visible object |
| 3 | find and approach an object |
| 4 | navigate to a distant reachable point |
| 5 | approach a door |
| 6 | open a door |
| 7 | open/cross a door |
| 8 | find and interact with an object |
| 9 | perform two interaction subgoals in sequence |

The curriculum manager uses simulator metadata to define rewards/targets. That does **not** mean all of that metadata is automatically given to the neural policy.

## Install

Create/activate a virtual environment with your desired PyTorch build, then:

```bash
python -m pip install -e . --no-deps
```

For tests:

```bash
python -m pip install pytest
pytest -q
```

If you need the upstream AI2-THOR controller package and do not already have a fork-specific installation:

```bash
python -m pip install ai2thor
```

## Validate the old 2-D path

```bash
continuous-brain \
  --state-dir state/test_v04 \
  --vision-backbone pixel-stats \
  --visual
```

Open `http://localhost:8765`.

## Run AI2-THOR

Set the executable explicitly:

```bash
export THOR_EXECUTABLE=/path/to/your/AI2-THOR-executable
```

Then:

```bash
continuous-brain \
  --environment thor \
  --scene FloorPlan1Discovery_physics \
  --thor-executable "$THOR_EXECUTABLE" \
  --state-dir state/embodied_brain_v041 \
  --curriculum-level 0 \
  --vision-backbone pixel-stats \
  --visual \
  --open-browser
```

Or:

```bash
LEVEL=0 ./run_thor.sh
```

## Bootstrap from a preserved v0.3 brain

Use a **new** state directory:

```bash
continuous-brain \
  --environment thor \
  --scene FloorPlan1Discovery_physics \
  --thor-executable "$THOR_EXECUTABLE" \
  --state-dir state/embodied_brain_v041 \
  --bootstrap-checkpoint /path/to/v0.3/state/my_brain/brain.pt \
  --curriculum-level 0 \
  --vision-backbone pixel-stats \
  --visual
```

This imports compatible neural tensors once but starts a fresh optimizer, counters and episodic memory for the embodied lifetime.

## Pretrained DINOv3

Once you have authorized/downloadable DINOv3 weights and Transformers/Pillow installed:

```bash
continuous-brain \
  --environment thor \
  --scene FloorPlan1Discovery_physics \
  --thor-executable "$THOR_EXECUTABLE" \
  --state-dir state/embodied_brain_v041 \
  --curriculum-level 0 \
  --vision-backbone dinov3-vits16 \
  --vision-every 5 \
  --visual
```

The foundation model is frozen. The small visual adapter and Continuous Brain remain trainable.

## Removing privileged navigation information

Early training:

```bash
--goal-signal teacher-nav --goal-dropout 0.0
```

Progressively harder:

```bash
--goal-signal teacher-nav --goal-dropout 0.25
--goal-signal teacher-nav --goal-dropout 0.50
--goal-signal teacher-nav --goal-dropout 0.80
```

Sensor/vision-only policy observation:

```bash
--goal-signal sensor-only
```

The trainer can still use hidden simulator targets for reward calculation. This is analogous to privileged training supervision: the policy itself does not receive the target vector in sensor-only mode.

## Sim-to-real robustness knobs

v0.4.1 continues the transfer work with:

```bash
--action-noise 0.05
--goal-dropout 0.25
--safety-distance 0.20
```

Future releases should add camera/lighting/material randomization, odometry noise, depth noise/dropout, asynchronous latency, spatial memory/SLAM, multiple simulators and a real-robot adapter.

## Curriculum runner

```bash
export THOR_EXECUTABLE=/path/to/executable
export SCENE=FloorPlan1Discovery_physics
LEVEL=0 GOAL_DROPOUT=0.0 ./scripts/run_thor_curriculum.sh
```

v0.4.1 intentionally does **not** blind-advance through L0→L9 after a fixed step count. Advance `LEVEL` only after the rolling graduation statistics are satisfactory. The same v0.4.1 state branch can then continue into the next level.

## Live monitor

The browser monitor now displays:

- top-down diagnostic trajectory/goal
- first-person RGB camera
- yaw and camera horizon
- depth sectors
- curriculum level and current target
- high-level action plus simulator primitive
- Q values and recurrent activity
- world-model surprise/loss
- curiosity and visual novelty
- replay and episodic/visual memory counts
- curriculum-valid actions (invalid Q rows are visibly masked)
- stagnation counter and rolling graduation statistics

## Safety / scope

This is research software. The v0.4.1 safety stop prevents a simulated `advance` intent when the center depth is below a configured threshold, but it is not a sufficient real-world safety layer. A physical embodiment needs independent actuator limits, collision avoidance, emergency stop, human override, hardware watchdogs and validation outside the learning policy.

## Next architectural targets

- egocentric/metric spatial memory and object permanence
- explicit depth/geometry encoder
- semantic visual goal embeddings instead of privileged point goals
- learned uncertainty / pose belief
- temporal video/world-model teacher integration
- hierarchical skill library above the five intent primitives
- cross-simulator adapters (Habitat / Isaac-style physics)
- domain randomization and real-sensor replay
- real robot adapter behind a hard non-learning safety controller
