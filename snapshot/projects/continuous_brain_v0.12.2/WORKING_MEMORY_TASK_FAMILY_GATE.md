# Continuous Brain — Varied Counterfactual Task-Family Gate

Status: **environment admission gate only; no neural training authorized by this file**

## Why this gate exists

The previous fixed `CuedDetourWorld` had only two effective trials at a given
corridor length: one fixed world with a LEFT cue and the same fixed world with
a RIGHT cue. Repeating nominal RNG seeds therefore produced pseudoreplication.

This replacement task family makes the independent experimental unit explicit:
a `condition_id` defines one relative delayed-decision problem, and each
condition is instantiated as a matched counterfactual pair:

- same geometry, dynamics, start, nuisance rendering, and delay;
- cue forced LEFT in one member;
- cue forced RIGHT in the other.

Formal evaluation uses the **condition pair** as the unit. An individual
condition passes only when the required criterion is satisfied on both members.

## Finite non-wrapping family

The first-generation family contains exactly **2,360 structurally unique
conditions**. IDs outside `[0, 2359]` raise an error rather than silently
wrapping to a previously used geometry.

Frozen split:

- train: `0..1599` — 1,600 unique structural conditions
- dev: `1600..1979` — 380 unique structural conditions
- test: `1980..2359` — 380 unique structural conditions

No structural condition appears in more than one split.

## Structural factors

Relative task structure varies across:

- corridor length: 8 levels
- horizontal goal offset: 5 levels
- vertical goal offset: 4 levels
- cue-visible duration: 4 levels
- movement step/dynamics: 4 levels

Only combinations with at least three fully cue-hidden pre-decision steps are
admitted.

Additional cue-independent sensory variation comes from:

- absolute x translation: 3 levels
- absolute y translation: 3 levels
- neutral visual distractor pattern: 4 levels

The diversity audit deliberately does **not** count those nuisance variations
as separate relative structural conditions.

## Counterfactual integrity contract

For every `condition_id`, LEFT and RIGHT members must satisfy all of these:

1. identical non-cue task specification;
2. identical non-cue initial state;
3. different cue-bearing initial observation/render;
4. identical teacher actions before the cue-dependent decision;
5. after cue disappearance, observation arrays are bit-identical between
   members until the first cue-dependent action;
6. after cue disappearance, rendered RGB arrays are bit-identical between
   members until the first cue-dependent action;
7. first cue-dependent teacher actions are LEFT for the left member and RIGHT
   for the right member;
8. the optimal teacher solves both members;
9. paired teacher path lengths are equal.

Any failure invalidates the generator before model training.

## Diversity admission thresholds

For an audited set of N conditions:

- spec hashes unique: 100%
- relative structural signatures unique: 100%
- conservative declared behavioral signatures unique: at least 70%
- conservative *realized* behavioral signatures unique: at least 70%
- cue-hidden sensory trace hashes unique: at least 90%
- all counterfactual-integrity checks: pass
- all teacher-pair success checks: pass

These thresholds are fixed before model training.

The behavioral signatures intentionally collapse absolute translation and
cosmetic distractors. This is the guard against "many coordinates, one
effective problem."

## Sandbox qualification before delivery

The generator/test/audit logic was executed in an isolated compatibility
sandbox before delivery.

First 256 train conditions:

- unique spec hashes: 256 / 256
- unique structural signatures: 256 / 256
- unique declared behavioral signatures: 230 / 256
- unique realized behavioral signatures: 203 / 256
- unique cue-hidden sensory traces: 256 / 256
- pair integrity: pass
- teacher pair success: pass

Full frozen dev range (`1600..1979`):

- unique structural signatures: 380 / 380
- unique declared behavioral signatures: 324 / 380
- unique realized behavioral signatures: 272 / 380
- unique cue-hidden sensory traces: 380 / 380
- gate: pass

Full frozen test range (`1980..2359`):

- unique structural signatures: 380 / 380
- unique declared behavioral signatures: 312 / 380
- unique realized behavioral signatures: 266 / 380
- unique cue-hidden sensory traces: 380 / 380
- gate: pass

These sandbox runs validate the environment logic itself. The real-project run
below is still required before any neural training.

## Required real-project gate

From the project root:

```bash
python -m continuous_brain.test_varied_cued_detour_world

python audit_varied_cued_detour_family.py \
  --start-id 0 \
  --conditions 256 \
  --output state/working_memory_m1/task_family_audit_train256.json

python audit_varied_cued_detour_family.py \
  --start-id 1600 \
  --conditions 380 \
  --output state/working_memory_m1/task_family_audit_dev.json

python audit_varied_cued_detour_family.py \
  --start-id 1980 \
  --conditions 380 \
  --output state/working_memory_m1/task_family_audit_test.json
```

All three audits must end with:

```text
TASK_FAMILY_ADMISSION: PASS
VARIED_CUED_DETOUR_FAMILY_AUDIT_COMPLETE
```

## What happens only after this passes

The next neural milestone is writable working memory around the protected 14k
InfantBrain. Its own DirectML qualification remains separate and must establish
exact no-op initialization, causal memory-input influence, full-BPTT gradients,
learned nonzero memory, exact historical-base preservation, and no reported CPU
fallback before a long training run.

The later model-level admission criterion is causal, not correlational:

> On genuinely held-out structural conditions, changing only the recalled
> memory content must predictably change the later cue-dependent decision.

Planned interventions include:

- memory enabled;
- memory forced to zero;
- LEFT/RIGHT memory content swapped within the same condition pair;
- recurrent hidden state reset while memory is preserved;
- both memory and hidden state reset.

A model does not pass because aggregate accuracy is high. It passes only if
memory content is demonstrably load-bearing under intervention.
