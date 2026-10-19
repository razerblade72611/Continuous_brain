# Continuous Brain v0.4.2

Continuous Brain is an experimental persistent embodied-agent architecture. It combines a continuously active recurrent neural state, online reinforcement learning, predictive world modeling, episodic/visual memory, intrinsic motivation, optional frozen pretrained perception/language models, and a simulator/robot embodiment boundary.

v0.4.2 is the **Hierarchical Navigation Correction**. It preserves the v0.4.1 neural dimensions and checkpoint compatibility while repairing the failure mode exposed by the first real Level-1 AI2-THOR run.

## Ultimate goal

The long-term goal is to build a **general, continuously operating multimodal embodied agent** that can eventually:

- inherit broad language, vision and video/world knowledge from strong pretrained foundation models rather than relearning those capabilities from scratch;
- maintain its own persistent recurrent state, autobiographical/episodic memory and spatial memory across time;
- continue learning from its own experience without routinely destroying previously mastered skills;
- reason and communicate at a level approaching leading LLM/VLM systems while adding capabilities those models do not inherently provide: lifetime adaptation, persistent personal experience, learned action consequences, hierarchical skills and physical embodiment;
- learn robust navigation and manipulation first in simulation, then across multiple simulators, and finally transfer the same high-level cognitive interfaces to real sensors and actuators;
- remain auditable and controllable, with real-world safety enforced independently of the learning policy.

The intended architecture is therefore **not** “train a frontier foundation model from random weights.” It is:

```text
pretrained language / vision / temporal teachers
                    │
                    ▼
            trainable adapters
                    │
                    ▼
        CONTINUOUS BRAIN CORE
 persistent state + world model + memory
                    │
          hierarchical skills
                    │
                    ▼
             embodiment API
                    │
       simulator(s) → real robot
```

## v0.4.2 architecture

```text
                       FINAL TASK GOAL
                              │
                   training-only path teacher
                              │
                              ▼
                       NEXT WAYPOINT
                              │
             ┌────────────────┼────────────────┐
             │                │                │
            RGB             depth          body state
             │                │                │
             ▼                │                │
     frozen visual model      │                │
      (optional DINOv3)       │                │
             │                │                │
             ▼                ▼                ▼
       visual adapter       embodiment sensor adapter
             │                │
             └────────┬───────┘
                      ▼
               persistent GRU
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
      policy       world model    memory
        │
        ▼
  idle / advance / turn L / turn R / interact
        │
        ▼
 embodiment executor + safety boundary
        │
        ▼
                AI2-THOR
```

The five-output policy head and six structured sensor slots remain unchanged so v0.4.1/v0.3 compatible weights can be retained.

## Why v0.4.2 exists

The first real L1 run exposed several problems:

1. L1 statistics inherited successful L0 outcomes, so the initial displayed success rate was not trustworthy.
2. L1 rewarded distance progress but did not reward a correct turn toward the goal, so the mastered L0 orientation primitive was not being composed cleanly with forward motion.
3. `advance` was newly unlocked at L1, but exploration continued from the already-decayed lifetime epsilon schedule.
4. A globally reachable final target can lie behind walls; a direct target bearing can therefore point into geometry before the agent has learned a map/planner.
5. Replay TD targets were masked using the currently active curriculum action set rather than the action set of the replayed skill.

v0.4.2 fixes all five.

## Level-local statistics

Curriculum evidence is now kept separately for every level. L0 outcomes no longer make a fresh L1 policy appear successful.

A migrated v0.4.1 checkpoint keeps its old aggregate counters as diagnostic legacy history, but **does not seed new v0.4.2 level-specific graduation windows** because the old window mixed levels.

## L1 reachable-waypoint teacher

During early `teacher-nav` L1 training, the environment builds a graph from AI2-THOR `GetReachablePositions` cells and supplies a short local waypoint rather than the final destination vector.

```text
FINAL ★
   │ wall
   │
   │      waypoint 2 ○
   │                │
   └────── ○ waypoint 1
           │
           A agent
```

The neural policy still sees only the same two normalized goal-vector slots. What changes is that those slots point to the **next collision-free teacher waypoint**.

Initial paths are deliberately short (`--nav-initial-path-steps 2`). As recent L1 success improves, the target generator expands toward `--nav-max-path-steps 8`.

This is privileged training supervision, not the intended final navigation system. Later versions will replace it with learned spatial memory/planning and progressively remove the goal signal.

## L1 compositional reward

L1 now combines:

```text
angular progress toward waypoint
+ distance progress toward waypoint
+ successful forward-motion bonus
+ waypoint completion bonus
- blocked/SafetyStop penalty
- wrong-way forward penalty
```

Useful turning resets the L1 stagnation watchdog even when Euclidean distance has not changed.

## New-level exploration reset

A curriculum-specific epsilon floor is added on top of the lifetime epsilon schedule. When L1 is first unlocked, exploration starts at approximately `0.35` by default and decays toward `0.08` over 5,000 L1 action steps.

This gives the newly available `advance` action enough opportunities to acquire useful Q-values.

## Skill rehearsal and tagged replay

By default, 15% of L1+ episodes rehearse previously learned curriculum levels:

```text
L1 training
├── ~85% L1 waypoint navigation
└── ~15% L0 orientation rehearsal
```

Replay transitions carry their skill level and legal-action mask. Training batches are biased toward the current skill while reserving capacity for older skills. TD bootstrap masking is performed **per replay row** using the replayed skill's legal actions.

This is the first explicit protection against catastrophic forgetting in the embodied curriculum.

## Curriculum action masks

| Level | Allowed intents |
|---:|---|
| 0 | idle, turn_left, turn_right |
| 1–5 | idle, advance, turn_left, turn_right |
| 6–9 | all five intents including interact |

The mask applies to epsilon exploration, greedy action selection, replay TD targets and optional slow planning.

## Curriculum

| Level | Task |
|---:|---|
| 0 | orient toward a goal direction |
| 1 | navigate through short reachable-waypoint paths |
| 2 | approach an initially visible object |
| 3 | find and approach an object |
| 4 | navigate to a distant reachable point |
| 5 | approach a door |
| 6 | open a door |
| 7 | open/cross a door |
| 8 | find and interact with an object |
| 9 | perform two interaction subgoals in sequence |

## Install / upgrade

Activate your existing venv and install the editable package:

```bash
python -m pip install -e . --no-deps
python -m pytest -q
```

The packaged test suite should report 28 passing tests.

## Recommended first v0.4.2 L1 run

Preserve the v0.4.1 diagnostic state and create a new branch from it rather than overwriting it.

```bash
export THOR_EXECUTABLE="$HOME/projects/ai2thor-humanoid/builds/thor-discovery-Linux64/thor-discovery-Linux64"

continuous-brain \
  --environment thor \
  --scene FloorPlan1Discovery_physics \
  --thor-executable "$THOR_EXECUTABLE" \
  --state-dir state/embodied_brain_v042 \
  --curriculum-level 1 \
  --goal-signal teacher-nav \
  --goal-dropout 0.0 \
  --stagnation-steps 60 \
  --rehearsal-fraction 0.20 \
  --nav-initial-path-steps 2 \
  --nav-max-path-steps 8 \
  --vision-backbone pixel-stats \
  --visual \
  --open-browser
```

For your first recovery run, `0.20` rehearsal is recommended because the v0.4.1 L1 experiment may have partially altered the mastered L0 policy. After L1 becomes stable, the default `0.15` is reasonable.

## Expected v0.4.2 telemetry

L1 terminal lines include both angular and translational evidence:

```text
brain=turn_right thor=RotateRight dDist=+0.000 dBearing=+30.0 move=0.000 wp=1/2
```

and:

```text
brain=advance thor=MoveAhead dDist=+0.250 dBearing=+0.0 move=0.250 wp=2/2
```

The monitor shows the final target separately from the active waypoint, the configured curriculum level vs rehearsal level, level-local graduation statistics and the new-level exploration floor.

## Graduation

L0 retains its strict median-success target (default 6 steps). L1 uses the same 95% rolling-success / <=2% timeout requirement, but its median-step allowance scales with configured path length because navigation necessarily requires more actions than pure orientation.

Do not advance levels based on wall-clock time or raw training steps alone.

## Pretrained perception

`pixel-stats` remains useful for validating the embodied control architecture. A frozen DINOv3 provider is also included. The longer-term plan is to use pretrained vision/video/language models as stable teachers or encoders and train small adapters plus the persistent Continuous Brain core.

## Removing privileged navigation information

Once L1 is robust under waypoint supervision:

```bash
--goal-dropout 0.10
--goal-dropout 0.25
--goal-dropout 0.50
```

Eventually:

```bash
--goal-signal sensor-only
```

That transition should coincide with learned spatial memory/planning rather than simply removing the teacher before the agent has an alternative representation.

## Safety / scope

This is research software. `SafetyStop` is a simulator guard, not a complete physical safety system. A real embodiment requires independent collision avoidance, actuator/velocity/force limits, emergency stop, hardware watchdogs, human override and validated fail-safe behavior outside the learning network.

## Next architectural targets

- persistent metric/egocentric spatial memory and object permanence
- learned pose belief / uncertainty instead of perfect simulator pose
- semantic visual-goal embeddings
- explicit local navigation planner learned from spatial memory
- temporal/video world-model teacher integration
- hierarchical skill library above primitive intents
- cross-simulator evaluation and domain randomization
- real-robot sensor/actuator adapter
