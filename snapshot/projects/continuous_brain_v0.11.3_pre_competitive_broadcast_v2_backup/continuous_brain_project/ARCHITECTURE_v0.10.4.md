# Continuous Brain v0.10.4 Architecture — Continual World-Memory Consolidation

v0.10.4 is a continual-learning repair for the child-owned consequence model. It follows the v0.10.3 blind P4b hierarchical-planning pass and the subsequent discovery that the older ForagingWorld predictive-dynamics gate no longer passed after the earlier DetourWorld world-only adaptation.

## What remains frozen

The following v0.10.3 organs are frozen bit-for-bit during v0.10.4 consolidation:

- ventral vision cortex and all parent heads;
- dorsal spatial cortex and child-owned target cortices;
- temporal cortex;
- recurrent cognitive nucleus/global workspace;
- language, motor, and value/care cortices;
- dedicated recurrent PlanningCortex;
- conservative Bellman planning Q head;
- temporal Option/Intention Cortex;
- all non-world buffers and statistical state.

Only `world.*` is optimizer-trainable.

This is important because the v0.10.3 hierarchical planner does not call the old world ensemble; it reasons in the dedicated `planning.*` latent and delegates primitive control to `options.*`. World consolidation therefore repairs older physical knowledge without modifying the neural machinery that passed P4b.

## Why ordinary `--world-only` replay was not enough

The original P4 detour adaptation trained the consequence model on one domain at a time. v0.10.4 changes the training distribution rather than adding capacity:

1. each update contains one batch from every rehearsal domain;
2. domain losses are averaged equally, independent of dataset size;
3. executed actions are balanced inside each domain before sampling;
4. the loss remains consequence-only: next spatial bus, structured next state, reward, success, and counterfactual action ranking;
5. the child-owned perception/nucleus path that supplies world-model inputs runs under `no_grad`;
6. every non-`world.*` tensor and buffer is compared bit-for-bit after training.

The v0.10.4 rehearsal rule is named:

`balanced-multidomain-action-rehearsal-v1`

## Fresh retention evidence

The old v0.9.5 P3 blind nursery is historical evidence and has already informed the v0.10.4 design. It is therefore not reused as the final blind qualification set.

v0.10.4 adds parent-free raw transition holdouts. A holdout contains only:

- current RGB observation;
- next RGB observation;
- physical context and next context;
- actual executed action;
- reward/success/collision outcome;
- episode continuity metadata.

No DINO, language parent, or child model is queried to create the retention evidence. A fixed collection behavior policy exists only to generate varied executed transitions; it is not stored as a supervision target.

Before consolidation, the two holdouts and the allowed training nurseries are sealed with a transition digest that includes `lessons.jsonl` and every referenced RGB image. `train-world-consolidation` verifies that seal before changing weights.

## Qualification

Fresh ForagingWorld and DetourWorld holdouts use the same predictive criteria established in the P3 line:

- enough causal/action-diverse transition evidence;
- next spatial prediction beats the no-change baseline by the fixed `0.98` gate and has positive wrong-action advantage;
- structured next-context prediction beats the no-change baseline by the fixed `0.80` gate and has positive wrong-action advantage;
- success prediction accuracy is at least `0.70`.

Both domains must pass.

## What this release does not claim

v0.10.4 release tests establish software isolation and the qualification protocol. They do not establish that consolidation has succeeded for David's persistent checkpoint until the fixed rehearsal run and newly sealed holdouts are actually evaluated. The project does not claim consciousness, sentience, personhood, or AGI.
