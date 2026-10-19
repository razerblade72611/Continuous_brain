# Continuous Brain v0.9.0 — Parented Development

## Thesis

The individual is a persistent integrated child architecture. Powerful pretrained models are external parents/teachers, not the child's neural identity. The child begins with randomly initialized cortices and learns through lived experience, demonstrations, teacher targets, correction, replay and consolidation. Parent authority is reduced only after competency gates pass.

This is an engineering architecture for a persistent artificial individual. It is not a scientific claim of consciousness or personhood.

## Structure

```text
                       EXTERNAL PARENTS
        DINOv3        Qwen3.5        V-JEPA 2       Whisper
          |              |              |              |
          | targets      | lessons      | temporal     | transcripts
          v              v              v              v
+----------------------------------------------------------------+
|                     DEVELOPING CHILD                            |
|                                                                |
| Raw Vision Cortex ----+                                        |
| Raw Audio Cortex -----+                                        |
| Raw Language Cortex --+--> STABLE LATENT BUS ----------------+ |
| Raw Temporal Cortex --+                                      | |
|                                                              v |
|              Persistent Recurrent Cognitive Nucleus            |
|                identity / goals / attention / routing           |
|                         |             |                          |
|                         v             v                          |
|                   Value Cortex     World Model Ensemble         |
|                         |             |                          |
|                         +-------> Motor Cortex -----> action     |
|                                                                |
| Episodic Memory <--> Semantic Memory <--> Procedural Memory     |
+----------------------------------------------------------------+
                              |
                              v
                            WORLD
```

Parents are outside the box.

## What is inherited vs individual

### External inheritance
- high-quality visual targets from DINOv3
- language/reasoning demonstrations from Qwen
- temporal representation targets from V-JEPA 2
- speech transcripts from Whisper
- optional stronger cloud teachers and evaluators

### Individual neural weights
- raw-pixel visual cortex
- byte-level transformer language cortex
- recurrent global-workspace nucleus
- temporal cortex
- action-conditioned world-model ensemble
- value/care cortex
- motor cortex
- all learned cortical interfaces

### Individual persistent state
- identity and lineage metadata
- recurrent hidden state
- autobiographical episodes
- semantic and procedural memories
- parent provenance and developmental history
- competency/weaning state

## Stable latent bus

The latent bus is the primary scaling contract. A cortex may grow or a parent may be replaced without changing the nucleus/memory protocol. Parent-specific projection heads are detachable scaffolding: they train a child cortex to reproduce useful parent representations, but the child cortex's own bus output is what survives at runtime.

## Parent weaning

Each capability has separate authority:

```text
parent authority 1.0 ------------------------------> 0.0
student authority 0.0 -----------------------------> 1.0
                    only after held-out passes
```

A child can therefore become autonomous in vision before language, or motor control before high-level reasoning. No global "parents off" switch is required.

## Memory and truth provenance

Memory kinds:
- episodic/autobiographical
- semantic
- procedural

Provenance:
- lived
- teacher
- imagined
- imported

Imagined futures must never be silently converted into lived memories.

## World model

The child's own action-conditioned ensemble predicts future latent state, reward and success. Ensemble disagreement is an epistemic-uncertainty signal. Foundation temporal models may teach representations, but they are not substituted for the child's body-specific world model.

## Continuity invariant

Fast recurrent state and each autobiographical episode are committed in one SQLite transaction. Model weights are versioned checkpoints; the runtime state records which checkpoint produced the experience.

## Scale profiles

Approximate trainable child parameters with a DINO-B parent head:
- proof: 7.5M
- balanced: 47.7M
- scale: 172.7M

These are proof/development sizes. The latent contract is designed to permit much larger language, world-model or sensory cortices after the architecture gates pass.
