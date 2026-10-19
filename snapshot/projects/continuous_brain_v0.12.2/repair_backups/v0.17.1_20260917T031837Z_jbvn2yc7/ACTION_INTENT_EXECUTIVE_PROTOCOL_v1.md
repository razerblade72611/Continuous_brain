# Continuous Brain v0.17.0 — Executive Action Intent Cortex

## Why this is a fundamental redesign

v0.16.0 established that recalled content is modular:

- donor consistency: 0.9838 / 0.9719 / 0.9706.

But ordinary own-content action control remained weak:

- pair-full: 0.8137 / 0.8387 / 0.8456;
- six-way: 0.7806.

Therefore memory is no longer the bottleneck. Workspace-delta steering is.

## New control boundary

The protected admitted brain computes:

`historical workspace + canonical recall -> frozen motor -> base action logits5`

The executive does NOT edit workspace.

It receives:

- canonical recall128;
- late rule;
- frozen base action logits5.

It emits:

- action-logit residual5.

Final action distribution:

`base_logits + executive_intent_delta`.

The world model and workspace remain untouched.

This creates a cleaner hierarchy:

`remembered content -> executive intent -> protected motor baseline`.

## Safety / causal isolation

The executive has:
- no raw memory64;
- no raw hidden state;
- no image/context input;
- no motor weight access.

Every feature path contains transformed canonical recall, so zero recalled
content produces exact zero intent delta.

RULE_NONE is an exact no-op.

## Training

v3 TRAIN only.

Every update contains:
- all six own cue×rule members;
- six coherent donor-response members.

Donor training holds recipient base logits fixed while substituting opposite-cue
canonical recall. Thus the executive explicitly learns portable intent.

Objective:
`0.5 own + 0.5 donor`.

Stages:
- update 1600 at 2e-4
- update 3200 at 1e-4 if needed
- update 4800 at 5e-5 if needed

Frozen gate:
- each rule pair-full >= 0.95
- each rule donor >= 0.95
- off <= 0.05
- zero-content <= 0.05
- six-way >= 0.90
- exact NONE no-op
- strong factor slices.

## After PASS

Build a brand-new v4 structural family and evaluate the frozen controller once.

If v4 passes, stop single-item rule assays and move to:
1. multi-item/order-sensitive memory;
2. goal-conditioned model-based planning;
3. metacognitive confidence;
4. long-horizon 3-D embodiment.

If v0.17.0 cannot pass TRAIN by 4800 updates, do not add another rule MLP.
The next redesign should replace fixed primitive-action supervision with an
option/action-intent hierarchy rather than continuing this assay family.
