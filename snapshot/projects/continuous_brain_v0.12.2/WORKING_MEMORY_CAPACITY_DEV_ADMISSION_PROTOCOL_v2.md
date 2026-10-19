# Continuous Brain — Reusable Working Memory DEV Admission v2

Status: **thresholds frozen before first DEV access for the capacity-ungated candidate**

Candidate:
- protected 14k historical brain;
- 64-D gated writer;
- memory-only nonlinear reader `64→256→128` with tanh and static learned scale;
- no contextual recall gate;
- no direct memory→motor shortcut.

Checkpoint:
`state/working_memory_recall_capacity_control_phase1_v1/latest.portable.pt`

DEV IDs: `1600..1979` (380 independent structural conditions).
TEST IDs: `1980..2359` remain sealed.

## Behavioral/causal admission

Required:
- DEV pair full-action >= 0.80;
- Wilson 95% lower bound >= 0.75;
- LEFT and RIGHT member full >= 0.85;
- member gap <= 0.10;
- pair side-sign >= 0.95;
- direct-recall pair-full gain over recall-off >= 0.50;
- paired one-sided discordance p < 1e-6;
- same-hidden direct-only pair side >= 0.95;
- hidden-reset direct-only pair side >= 0.75;
- hidden-reset swapped-memory donor-consistency >= 0.75;
- direct memory-swap mean > 0;
- bootstrap 95% lower bound > 0;
- positive swap fraction >= 0.90;
- sign p < 1e-6;
- all-zero counterfactual logits equal within 1e-6.

Every factor level must additionally have:
- pair full >= 0.60;
- pair full no more than 0.20 below overall DEV;
- LEFT/RIGHT member gap <= 0.20.

Factor slicing includes structural variables plus start translation and distractor.
`start_x` and `start_y` remain known coupled nuisance factors and are reported,
not interpreted as separable causal effects.

## Reusable-representation admission on DEV

Brand-new readers are fit ONLY on frozen TRAIN memories and evaluated on frozen
DEV memories. No DEV fitting is permitted.

Required:
- linear cue reader DEV accuracy >= 0.95;
- every reported corridor-length/cue-step DEV slice >= 0.90;
- post-write NORMAL/INVERT reader overall >= 0.95;
- NORMAL >= 0.95;
- INVERT >= 0.95;
- both instructions correct for the same stored memory >= 0.90;
- memory-zero control <= 0.60;
- forced-NORMAL instruction control <= 0.60;
- instruction-flip counterfactual >= 0.95;
- swapped-memory donor consistency >= 0.95;
- concatenated linear interaction control <= 0.60;
- factorized cue-reader + late instruction >= 0.95.

## Interpretation

A PASS means the reusable working-memory mechanism has survived a genuinely
held-out DEV split without architectural or threshold changes.

It does not yet authorize a consciousness claim and does not itself open TEST.

After a PASS:
1. freeze this architecture/checkpoint/evaluator definition;
2. promote the mechanism into the next main brain development branch;
3. move active engineering to executive composition, planning, world-model use,
   metacognition, and embodiment;
4. keep TEST sealed until the broader next-stage architecture has a justified
   final validation point.
