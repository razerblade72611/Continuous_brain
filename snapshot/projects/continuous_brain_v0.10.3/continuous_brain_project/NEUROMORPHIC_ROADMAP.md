# Continuous Brain Neuromorphic / Spiking Roadmap

## Current status

Continuous Brain v0.5.3 is **not** a spiking neural network. Its trainable cognitive core uses conventional PyTorch recurrent/dense neural computation, currently centered on a recurrent GRU-style state plus learned prediction/action heads. This was intentional: it preserves straightforward gradient training, checkpoint migration, and compatibility with pretrained dense LLM/VLM/vision representations.

## Why spiking still fits the original goal

Continuous Brain is intended to operate continuously, maintain internal state, react to events, and ultimately run on embodied hardware with finite energy. Spiking neural networks (SNNs) are therefore a serious architectural research direction, especially for fast temporal processing and sparse event-driven computation.

The project should test SNNs where their inductive bias is strongest rather than replacing the entire intelligence stack prematurely.

## Target hybrid architecture

```text
Pretrained LLM / VLM / video models
              |
       dense semantic state
              |
     slow cognitive core
 memory / planning / world model
              |
      +-------+-------+
      |               |
      v               v
spiking fast loop   skill system
      |               |
 salience             actions
 novelty               |
 temporal events       v
 reflex gating      embodiment
 sensor fusion
```

### Dense / slow system

Use conventional neural models for:
- pretrained language and vision interoperability;
- semantic concepts and abstract reasoning;
- large associative memories;
- explicit planning and world-model rollouts;
- skill selection;
- adapter learning from foundation models.

### Spiking / fast system

Experiment with SNN modules for:
- temporal novelty and salience;
- event-camera streams;
- audio/event onset processing;
- low-latency reflex gating;
- sensor fusion;
- continuous homeostatic dynamics;
- sparse working-state maintenance;
- timing-sensitive motor primitives.

## Phased implementation

### Phase S0 — software research harness
Add an optional PyTorch SNN module using differentiable surrogate-gradient training. It must be completely optional and must not change existing checkpoint tensor shapes when disabled.

Success criterion: on synthetic temporal/event tasks, the SNN sidecar should learn a useful signal that is not trivially reproduced by the existing scalar novelty heuristic.

### Phase S1 — embodied salience sidecar
Feed frame-change, depth-validity, collision, reward-surprise, audio/event, and proprioceptive deltas into a small recurrent spiking population. Produce compact outputs such as:
- salience;
- novelty;
- urgency;
- sensor-consistency confidence.

The dense brain consumes those values; the SNN does not control the robot directly.

### Phase S2 — temporal sensor processing
Where appropriate, move high-rate sensor preprocessing from frame-by-frame dense updates to event-based processing. Event cameras are especially relevant here because their output is already sparse and asynchronous.

### Phase S3 — spiking motor / reflex loop
Introduce a constrained fast spiking controller below deliberative cognition. It may influence low-level timing or reflex suppression, but remains underneath an independent physical safety supervisor.

### Phase S4 — neuromorphic backend
Only after software SNN modules demonstrate measurable value should we target neuromorphic hardware. Hardware choice must follow the learned workload rather than dictate the architecture prematurely.

## Evaluation requirements

An SNN module is adopted only if it wins on at least one meaningful metric without causing unacceptable regressions:
- temporal prediction quality;
- sample efficiency;
- event latency;
- robustness to irregular timing;
- compute utilization;
- measured energy on target hardware;
- continual-learning stability.

"More biologically inspired" is not itself a success criterion.

## Relationship to the Ultimate Goal

Spiking computation is a possible implementation mechanism for parts of Continuous Brain, not the project's identity. The project's defining properties remain:
- persistent lifetime state;
- continual learning;
- autobiographical, semantic, and spatial memory;
- learned world prediction;
- hierarchical reusable skills;
- strong pretrained language/vision knowledge;
- self-directed planning and exploration;
- simulation-to-real transfer;
- independent hard physical safety.
