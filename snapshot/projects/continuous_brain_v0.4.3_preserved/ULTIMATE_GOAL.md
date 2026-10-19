# Continuous Brain — Ultimate Goal

Build a persistent, continuously learning, multimodal embodied intelligence that combines the strongest reusable knowledge from pretrained foundation models with a distinct lifetime-learning architecture.

## Target capabilities

1. **Foundation-level perception and language** — leverage pretrained LLM/VLM/video models so the project inherits broad world knowledge instead of retraining it from scratch.
2. **Persistent internal state** — maintain recurrent state across decisions rather than treating every prompt/frame as a new session.
3. **Autobiographical and spatial memory** — remember events, places, objects, outcomes and learned skills across long time horizons.
4. **Continual learning** — adapt online while protecting previously acquired competencies from catastrophic forgetting, including across process restarts.
5. **World modeling and planning** — predict consequences of candidate actions and use internal simulation to choose behavior.
6. **Hierarchical skills** — compose orientation, navigation, interaction and manipulation primitives into long-horizon goals.
7. **Embodiment** — train in increasingly realistic 3-D worlds, transfer across simulators, then connect the same cognitive interface to real sensors and actuators.
8. **Leading multimodal reasoning quality** — approach the practical language/vision/reasoning usefulness of leading LLM/VLM systems by incorporating their pretrained strengths while retaining Continuous Brain's persistent memory, lifetime adaptation and embodied action loop.
9. **Independent safety and control** — keep actuator safety, emergency stop, limits, watchdogs and human override outside the learning policy.
10. **Evidence-based capability promotion** — skills graduate only from clean evaluation evidence, with recoverable mastered snapshots and cross-scene validation before harder curricula depend on them.

## Core architectural bet

```text
PRETRAINED GENERAL KNOWLEDGE
LLM + VLM + temporal/world teachers
              │
              ▼
       small adapters
              │
              ▼
     CONTINUOUS BRAIN
 persistent recurrent cognition
 memory + world model + drives
              │
              ▼
      hierarchical skills
              │
              ▼
       embodiment layer
              │
      ┌───────┼────────┐
      ▼       ▼        ▼
 AI2-THOR  Habitat   physics sim
              │
              ▼
          real robot
```

The project should not spend resources recreating frontier-scale pretraining when strong pretrained networks already exist. Cloud GPU spending should primarily target adapters, temporal/world-model learning, parallel embodied experience, skill learning and teacher-to-student distillation.

## Development rule

A curriculum feature is not considered learned merely because training reward rises. Promotion should require:

- clean per-skill success metrics;
- no privileged-information leak beyond the explicitly permitted teacher signal;
- real action execution rather than tolerance-radius shortcuts;
- retention of earlier skills;
- reproducible checkpoint/snapshot recovery;
- progressively harder scenes and sensor perturbations;
- eventually, transfer to an embodiment that does not expose simulator-only state.
