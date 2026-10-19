# Continuous Brain — Ultimate Goal

## Mission

Build a persistent, continuously learning, multimodal embodied intelligence whose broad language/vision/reasoning competence can inherit the best available pretrained foundation models, while the Continuous Brain architecture supplies persistent lifetime cognition, memory, world modeling, skill learning and embodiment.

The project is **not** trying to reproduce frontier-scale LLM/VLM pretraining from random weights. The intended system is:

```text
strong pretrained foundation intelligence
                  +
persistent recurrent lifetime state
                  +
autobiographical / semantic / spatial memory
                  +
online adaptation and world prediction
                  +
hierarchical embodied skills
                  +
independent physical safety layer
```

## Desired properties

A mature system should be able to:

- maintain persistent internal context beyond one prompt/session;
- perceive through vision, depth, audio, proprioception and other robot sensors;
- remember episodes, objects, places, people and task history;
- build and revise spatial/semantic world models;
- predict consequences of actions;
- plan using learned skills rather than issuing raw motor commands from an LLM;
- learn new skills while retaining older ones;
- recognize uncertainty and request/reacquire information;
- transfer from simulation to physical embodiment;
- use strong pretrained LLM/VLM/video models as teachers, representational priors or planners without making them the sole persistent cognitive substrate.

## Safety / personhood stance

High-capability embodied systems should be developed with auditable safety controls, explicit capability boundaries and careful governance. Physical actuators must remain behind independent non-learning safety constraints, watchdogs and emergency-stop mechanisms.

The architecture should not assume that a learned agent is conscious or a person merely because it is persistent or sophisticated. Questions about consciousness/personhood require evidence rather than architectural labels.

## Roadmap

### v0.1–v0.3 — persistent learning core

Recurrent state, replay, prediction, episodic memory, intrinsic motivation, visualization and pretrained visual-representation boundary.

### v0.4.x — embodiment and hierarchical movement

Real AI2-THOR integration, curriculum action masks, mastered-skill snapshots, persistent replay, L0 orientation and L1 multi-waypoint navigation.

### v0.5.x — spatial cognition (current phase)

- persistent occupancy/spatial memory;
- pose-belief interface;
- self-generated waypoints;
- object/landmark memory;
- teacher → internal-map transition;
- later visual/depth odometry, loop closure and uncertainty.

### v0.6 — hierarchical semantic skills

Reusable approach/open/pickup/cross/search/manipulate skills, explicit skill preconditions/outcomes, skill graph and continual rehearsal.

### v0.7 — language/VLM cognition

Foundation-model semantic perception and language reasoning connected to the persistent brain through constrained plans/subgoals, not direct raw actuator control.

### v0.8 — temporal world model

Video/temporal foundation representations, action-conditioned latent prediction, imagination/rollouts and distillation into the persistent adaptive brain.

### v0.9 — cross-simulator / sim-to-real

AI2-THOR → other 3-D simulators, domain randomization, sensor noise, dynamics randomization and explicit uncertainty.

### v1.0 — physical embodiment

Camera/depth/IMU/odometry/joints/touch/audio behind a hard safety supervisor with limits, watchdogs and emergency-stop mechanisms.

## Current north-star test

The decisive architectural transition now is:

```text
simulator provides waypoint
          ↓
brain follows waypoint
```

becoming:

```text
sensors + memory + internal map
          ↓
brain generates waypoint
          ↓
mastered navigation skill executes it
```

That is the purpose of v0.5.

## Neuromorphic / spiking research path

Continuous Brain is not committed to one neuron implementation. The current dense recurrent core is retained for pretrained-model interoperability and stable learning. A hybrid spiking path will be evaluated for event-driven temporal processing, salience, sensor fusion, homeostasis, and low-latency motor/reflex dynamics. Neuromorphic hardware becomes a deployment target only after software SNN modules demonstrate measurable benefit. See `NEUROMORPHIC_ROADMAP.md`.

## Infrastructure exit rule

Simulation infrastructure is considered sufficient when faults can be detected, isolated from learning, recovered or failed safely, and measured. Perfect simulator behavior is not a prerequisite for advancing the cognitive architecture. After spatial cognition is experimentally usable, priority moves to semantic skill composition, foundation-model grounding, richer world prediction, and cross-environment transfer.
