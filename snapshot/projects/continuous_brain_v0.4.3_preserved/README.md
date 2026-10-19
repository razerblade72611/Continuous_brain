# Continuous Brain v0.4.3

Continuous Brain is an experimental persistent embodied-agent architecture. It combines a continuously active recurrent neural state, online reinforcement learning, predictive world modeling, episodic/visual memory, intrinsic motivation, optional frozen pretrained perception/language models, and a simulator/robot embodiment boundary.

v0.4.3 is the **Navigation Bootstrap Correction**. It preserves the v0.4.x 6-input / 5-intent neural interface and checkpoint compatibility while making Level-1 navigation much harder to satisfy accidentally and much safer to iterate on.

## Ultimate goal

Build a persistent, continuously learning, multimodal embodied intelligence that inherits broad language/vision/world knowledge from strong pretrained foundation models and adds capabilities that a stateless prompt-response system does not inherently provide:

- persistent recurrent internal state;
- autobiographical, visual and spatial memory;
- online lifetime adaptation without routinely destroying old skills;
- learned action consequences and action-conditioned world prediction;
- hierarchical skill composition;
- long-horizon planning;
- simulation-to-real embodiment through a stable sensor/action interface;
- independent, auditable actuator safety outside the learned policy.

The intended end state is **foundation-model perception/reasoning + Continuous Brain lifetime cognition + embodiment**, not an attempt to recreate frontier-scale pretraining from random weights.

## Why v0.4.3 exists

The real v0.4.2 L1 run exposed four important failure modes:

1. a 0.25 m one-cell target could fall inside the old 0.35 m success radius before any translation occurred;
2. the simulator depth pre-stop could reject `MoveAhead` even when the teacher waypoint was a reachable adjacent cell;
3. the newly unlocked `advance` action still needed stronger composition with the already-mastered orientation primitive;
4. a nominal reachable-grid edge can disagree with the custom Unity build at runtime and needs online failure/replanning evidence.

v0.4.3 fixes those issues and adds restart/rollback defenses for future curriculum work.

## L1 now requires real locomotion

Defaults:

```text
grid movement             0.25 m
minimum teacher path      2 moves
navigation success radius 0.12 m
required real translations >= 1
```

A target cannot be counted as an L1 success just because it began one grid cell away. The episode must contain an actual successful translation.

## Orientation-gated forward motion

During early `teacher-nav` L1 training:

```text
|waypoint bearing| > 15°
        ↓
ADVANCE masked
idle / turn_left / turn_right available

|waypoint bearing| <= 15°
        ↓
ADVANCE unlocked
```

This deliberately composes the mastered L0 skill before asking the network to learn locomotion. The dynamic action mask is stored on every replay transition and therefore also applies to TD targets.

Later, once spatial memory replaces the privileged teacher, this gate can be reduced or disabled.

## Simulator-native collision learning

The old pre-emptive center-depth `SafetyStop` is disabled by default for simulator training. `MoveAhead` is sent to AI2-THOR and the simulator's actual success/failure and measured displacement become the learning evidence.

Depth is still logged and fed to the observation. The old guard remains available with:

```bash
--depth-safety-training
```

Do **not** interpret simulator-native collision training as a real-robot safety design. Physical hardware must retain an independent safety supervisor, emergency stop, force/velocity limits and watchdogs outside the policy.

## Reachable-edge learning and replanning

The L1 teacher starts from the reachable-position graph, but v0.4.3 no longer assumes every nominal neighbor remains valid forever.

If an aligned `MoveAhead` fails:

```text
current cell → next waypoint
       │
       └── MoveAhead fails
               ↓
        record edge failure
               ↓
        exclude it immediately
        for this episode
               ↓
        replan to final target
```

Repeated failures persistently blacklist the edge. Successful traversal clears stale failure evidence.

The edge-failure map is checkpointed.

## Honest per-skill statistics

L0/L1/... statistics remain separate. L1 success requires physical translation, and the monitor reports:

- current-level goals;
- successful translations;
- blocked advances;
- replans;
- persistently blocked edges;
- route-unavailable failures;
- success/timeout/blocked-move rates.

L1 graduation additionally rejects a policy with an excessive blocked-move rate (default maximum 10%).

## Skill rehearsal and persistent replay

Transitions remain tagged by skill level and legal action mask. v0.4.3 also persists a bounded replay cache across process restarts.

The persisted replay is **stratified**: it reserves recent rows from every represented skill before filling the remainder with the newest experience. A long L1 run therefore cannot silently evict every L0 rehearsal transition from the restart checkpoint.

Default persisted capacity:

```bash
--replay-checkpoint-items 8000
```

Replay is stored separately in `state/.../replay.pt`.

## Automatic mastered-skill snapshots

When a configured curriculum level satisfies its graduation criteria, v0.4.3 automatically writes:

```text
state/<brain>/skills/L0_mastered.pt
state/<brain>/skills/L0_mastered.json
state/<brain>/skills/L1_mastered.pt
...
```

A snapshot contains compatible neural weights, target network, optimizer state, curriculum evidence and a bounded embedded replay slice. It can be used as a later neural bootstrap or copied into a fresh branch as `brain.pt` for skill rollback testing.

## Crash-safe checkpoints

Normal checkpoint writes are atomic. Before replacing the primary checkpoint, the previous valid checkpoint is retained as:

```text
brain.previous.pt
```

If `brain.pt` becomes unreadable, startup automatically falls back to `brain.previous.pt`.

## Curriculum action masks

| Level | Allowed intents |
|---:|---|
| 0 | idle, turn_left, turn_right |
| 1 | dynamic: turn/idle until aligned; then idle, advance, turn_left, turn_right |
| 2–5 | idle, advance, turn_left, turn_right |
| 6–9 | all five intents including interact |

## Install

```bash
python -m pip install -e . --no-deps
python -m pytest -q
```

The packaged v0.4.3 suite should report **39 passed**.

## Recommended clean canonical restart

The v0.4.1/v0.4.2 navigation runs accumulated thousands of updates under known-bad L1 curricula. Preserve them as diagnostics, but do not use them as the canonical v0.4.3 navigation brain.

Bootstrap a fresh embodied branch from the preserved v0.3 neural brain and retrain the now-correct L0 once:

```bash
export THOR_EXECUTABLE="$HOME/projects/ai2thor-humanoid/builds/thor-discovery-Linux64/thor-discovery-Linux64"

continuous-brain \
  --environment thor \
  --scene FloorPlan1Discovery_physics \
  --thor-executable "$THOR_EXECUTABLE" \
  --state-dir state/embodied_brain_v043_clean \
  --bootstrap-checkpoint \
    ~/projects/continuous_brain_v0.3_preserved/state/my_brain/brain.pt \
  --curriculum-level 0 \
  --goal-signal teacher-nav \
  --goal-dropout 0.0 \
  --stagnation-steps 60 \
  --vision-backbone pixel-stats \
  --visual \
  --open-browser
```

Continue until the terminal prints a line similar to:

```text
[skill-mastered] L0 snapshot saved: .../skills/L0_mastered.pt
```

Then stop cleanly with `Ctrl+C`.

## Corrected L1 run

Continue the **same** state directory without `--bootstrap-checkpoint`:

```bash
continuous-brain \
  --environment thor \
  --scene FloorPlan1Discovery_physics \
  --thor-executable "$THOR_EXECUTABLE" \
  --state-dir state/embodied_brain_v043_clean \
  --curriculum-level 1 \
  --goal-signal teacher-nav \
  --goal-dropout 0.0 \
  --stagnation-steps 60 \
  --rehearsal-fraction 0.20 \
  --nav-min-path-steps 2 \
  --nav-initial-path-steps 2 \
  --nav-max-path-steps 8 \
  --nav-success-radius 0.12 \
  --nav-alignment-gate-degrees 15 \
  --vision-backbone pixel-stats \
  --visual \
  --open-browser
```

During early L1, look for a repeating composition such as:

```text
brain=turn_right thor=RotateRight ...
brain=advance    thor=MoveAhead dDist=+0.250 move=0.250 ok=1 ...
```

and verify that `L1goals` rises only after real translations.

## Helper scripts

- `scripts/run_l0_clean.sh` — clean L0 bootstrap from the preserved v0.3 brain.
- `scripts/run_l1_v043.sh` — corrected L1 continuation.
- `scripts/branch_from_skill_snapshot.sh` — create a new experimental branch from a mastered-skill snapshot.
- `scripts/run_thor_curriculum.sh` — generic curriculum launcher.

## Pretrained perception

`pixel-stats` remains the fastest control-path validation backbone. The frozen DINOv3 provider remains available. Once navigation mechanics are stable, the next perception step is to run the same curriculum with DINOv3 embeddings while keeping the large backbone frozen and training only the adapter/recurrent system.

## Removing privileged navigation information

Do not remove the teacher until L1 is genuinely reliable. Then progressively test:

```bash
--goal-dropout 0.10
--goal-dropout 0.25
--goal-dropout 0.50
```

Eventually:

```bash
--goal-signal sensor-only
```

That transition should coincide with the v0.5 spatial-memory/map system so the brain has an internal navigation representation to replace the teacher.

## Next architectural targets

The next major release should focus on **spatial cognition**, not more reward patches:

- persistent egocentric/metric 3-D spatial memory;
- pose belief and uncertainty rather than perfect simulator pose;
- object permanence and semantic landmarks;
- learned local-map construction;
- self-generated waypoints and route planning;
- path replanning from memory when geometry changes;
- evaluation-only runs with exploration disabled;
- cross-scene/cross-simulator validation before claiming transfer;
- domain randomization and sensor-noise curriculum;
- later V-JEPA/video-world-model teacher integration;
- real-robot adapter with an independent safety supervisor.
