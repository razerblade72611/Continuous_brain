# Continuous Brain v0.7.1 — Grounded Dialogue

## Purpose

Move from four narrow C1 utterance families toward honest, state-grounded dialogue while preserving the same persistent individual.

## Changes

- adds a learned seven-way dialogue-intent head
- adds explicit unknown-question rejection and confidence gating
- grounds identity in the persistent `SelfModel.name`
- grounds goal/state and next-step reports in the current persistent intention
- grounds temporal reports in the latest real lifetime experience metadata
- adds dialogue curriculum augmentation: paraphrases, identity, unknown/OOD and temporal Q/A
- adds runtime-compatible temporal nursery context using the existing 8-value memory tail
- adds `clone-dataset` and `prepare-dialogue` commands
- supports bootstrapping v0.7.1 training from a qualified v0.7.0 semantic cortex
- ships the tolerant local Qwen2-VL 2B visual-teacher path discovered during the first local training run
- ships the trusted teacher guard: VLM teaches vision; persistent state/curriculum supplies goal and stop/respect truth
- pins Transformers to `>=4.56,<5` and Hugging Face Hub `<1.0` in semantic/vision extras to avoid the incompatible Transformers 5.x transition seen in the first-conversation environment
- adds v0.7.1 qualification metrics for intent accuracy, unknown-intent accuracy and held-out OOD rejection
- continuation attachment archives the previous semantic cortex and preserves identity/executive/motor/memory/world state

## Important limits

- unknown rejection is learned within a deliberately small dialogue ontology; it is not broad natural-language understanding
- identity/current-goal answers use trusted grounding bridges; they are not evidence of self-awareness
- current care training is a first social/value curriculum, not a complete theory of caring or morality
- temporal dialogue currently reports the most recent real skill/outcome; rich autobiographical narrative memory is a later milestone
- no consciousness claim is made
