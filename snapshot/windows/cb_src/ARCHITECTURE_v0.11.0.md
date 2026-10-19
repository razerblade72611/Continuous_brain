# Continuous Brain v0.11.0 Architecture

## Purpose

v0.11.0 is the first capacity-growth release after the v0.10.4 child satisfied the engineering gates for large-scale training. It does **not** replace the child with a larger randomly initialized network. Instead, it embeds the complete v0.10.4 computation unchanged inside a larger trainable system.

This is an engineering continuity mechanism. It is not evidence of consciousness, personhood, or AGI.

## Function-preserving growth

Each major organ receives one or more residual growth stacks:

```text
legacy signal x
      |
      +-------------------------------+
      |                               |
      |       new growth branch       |
      |   Linear -> GELU -> Linear    |
      |                    ^          |
      |             zero initialized  |
      |                               |
      +---------- x + growth(x) ------+
```

The final projection of every new block is initialized to exact zeros. Therefore `growth(x) == 0` at the migration instant and every stack is exactly the identity map.

The expansion command separately verifies:

1. every legacy state tensor is copied exactly;
2. every new missing key belongs to a growth module;
3. audited outputs of the pre-growth and expanded models are exactly equal;
4. the expanded child exceeds 50M trainable parameters.

The default `balanced-v110` profile is approximately:

- pre-growth child: ~9.2M trainable parameters;
- new growth capacity: ~44.4M trainable parameters;
- expanded child: ~53.5M trainable parameters, plus small parent-head differences from the actual lineage;
- EMA target cortices are shadow copies and are not counted as independently trainable intelligence.

## Where capacity is added

Growth branches are integrated into:

- ventral raw-pixel vision;
- dorsal spatial vision;
- audio frontend output;
- fused visual bus;
- recurrent nucleus workspace;
- byte-level language hidden stream;
- semantic temporal bus;
- action-conditioned dynamics temporal bus;
- each world-model ensemble member;
- recurrent planning latent;
- option/intention latent processing;
- value/care pathway;
- motor pathway.

Some new capacity, especially audio and semantic-temporal growth, remains mostly dormant until corresponding lived curricula are available. v0.11.0 does not fabricate supervision merely to make every new parameter move.

## Growth-first routed training

The first large-scale phase intentionally freezes all legacy tensors. Only names belonging to growth modules are trainable. The deterministic route schedule is:

```text
perception -> language -> world -> embodiment -> planning
-> perception -> language -> planning -> world -> options -> repeat
```

The routes use different evidence and objectives:

- **perception:** replayed developmental image/language/body records; vision, spatial, fusion and nucleus growth paths;
- **language:** grounded Q/A rehearsal; language and nucleus growth paths;
- **world:** parent-free lived transitions, action-balanced within the selected environment; world and dynamics growth paths;
- **embodiment:** motor/value rehearsal where evidence is eligible;
- **planning:** contiguous lived multi-step sequences and recurrent self-rollout losses;
- **options:** the same anonymous option-discovery process used by v0.10.2, never wall/gap semantic labels.

The optimizer may contain all growth parameters, but route masks determine which growth paths are eligible to receive a gradient on each update.

## Rehearsal protection

v0.11.0 uses four levels of protection:

1. old parameters and buffers are frozen;
2. prior datasets remain in the training mixture;
3. every fifth routed update adds a child-only preservation loss against a frozen pre-growth reference reconstructed from the same legacy tensors (vision, spatial, fusion, workspace, language, motor, values, planning and option interfaces; the world model is deliberately excluded so new physics can consolidate);
4. the run ends with a bit-exact legacy-state comparison.

The first phase therefore tests whether **new capacity can learn without rewriting old capacity**. Controlled legacy-weight plasticity is intentionally deferred until the growth-first phase passes retention gates.

## Generalized embodied curriculum

Two parent-free environments preserve the same five primitive actions while increasing layout diversity.

### RandomizedDetourWorld

Each episode randomizes:

- barrier horizontal position;
- opening vertical position;
- opening size;
- whether the agent starts left or right of the barrier;
- start and target positions.

### TwoBarrierWorld

Two barriers appear with separate randomized openings. The default course places one opening high and the other low, requiring multiple temporal phases.

A geometry-aware collection policy may be used to obtain successful lived training trajectories. Those records set `motor_valid=false`; the collector is not installed as the child's motor policy. Evaluation is child-only.

## Colab checkpointing and resume

`train-whole-brain` writes the current `infant_training.pt` plus periodic snapshots:

```text
infant_training.v110_route_0000500.pt
infant_training.v110_route_0001000.pt
...
```

Metadata stores the cumulative routed step count and deterministic route schedule. A restarted run resumes the growth optimizer when compatible and continues route indexing rather than silently restarting the developmental schedule.

## Qualification philosophy

v0.11.0 deliberately separates four claims:

1. **migration:** exact function preservation;
2. **capacity:** >50M trainable parameters exist;
3. **training integrity:** legacy tensors stay unchanged during growth-first learning;
4. **capability:** old retention gates and new generalized-environment tests must be run after training.

Passing the first three does not automatically establish the fourth.
