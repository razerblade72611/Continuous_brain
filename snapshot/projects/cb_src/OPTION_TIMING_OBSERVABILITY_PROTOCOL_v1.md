# Continuous Brain v0.18.1 — Option Timing Observability Diagnostic

## Why this diagnostic exists

v0.18.0 separated executive control into:

- WHAT: canonical recalled content + late rule -> action intent
- WHEN: protected workspace + frozen base logits -> scalar execution gate

At update 4800 the WHAT/response side is essentially solved on v3 TRAIN:

- invert_lr pair-full 0.9912
- vertical pair-full 0.9906
- vertical_reverse pair-full 0.9931
- six-way 0.9862
- donor consistency 0.9988 / 0.9938 / 0.9906

But timing remains poor:

- pre-response action preservation 0.5630
- pre-response gate-off rate 0.5550
- response gate-on rate 0.8984

Therefore the next question is not "make another gate."
It is: **is response timing observable from the state we gave the gate?**

## No brain training

This package never updates the Continuous Brain.
It freezes the admitted brain and the v0.18 update-4800 checkpoint.

Only tiny CPU diagnostic readers are fit.

v3 DEV and TEST are not touched.

## Candidate timing representations

At every post-instruction timestep we collect:

1. raw observation;
2. context tensor;
3. recurrent hidden entering the current step;
4. protected post-core hidden for the current step;
5. historical workspace;
6. frozen base motor logits;
7. historical workspace + base logits (the actual v0.18 gate input);
8. post-core hidden + workspace + logits.

The existing trained v0.18 gate is measured as a reference.

## Anti-leak design

Cross-validation splits by CONDITION ID.

Both cue members and every timestep of one condition stay in the same fold.

Probe fitting uses a balanced boundary dataset:
- final pre-response state;
- response state.

The resulting classifier is additionally tested against ALL earlier
post-instruction pre-response states.

Cue-pair prediction gaps are reported to detect a timing reader that is
accidentally exploiting remembered cue identity.

## Frozen observability threshold

A representation is considered sufficient for a static WHEN controller only if
its held-out MLP reader reaches all of:

- boundary accuracy >= 0.95
- all-pre gate-off rate >= 0.95
- response gate-on rate >= 0.95

For protected hidden to be recommended, mean left/right cue-pair probability
gap must also be <= 0.05.

## Interpretation

A. workspace+logits passes:
   v0.18 had an optimization/supervision problem.

B. workspace+logits fails, protected hidden passes:
   timing state exists in the brain but was omitted from the gate.
   Next architecture should let only the scalar WHEN gate read protected hidden;
   WHAT remains isolated on canonical recalled content + rule.

C. raw observation passes but protected neural state fails:
   the lower state encoder is dropping task-progress information.

D. nothing static passes:
   stop static gate variants and add a recurrent option-state/progress controller.

No matter which outcome occurs, the working-memory substrate and WHAT pathway
are not reopened by this diagnostic.
