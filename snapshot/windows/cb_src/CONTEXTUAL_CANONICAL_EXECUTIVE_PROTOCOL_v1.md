# Continuous Brain v0.16.0 — Contextual Canonical Executive

## What v0.15.0 established

The richer content/rule compositor was a substantial improvement over v0.14.x,
but it still failed the fixed-final v3 TRAIN gate after 4,800 updates:

- invert_lr pair-full: 0.8706
- vertical pair-full: 0.9006
- vertical_reverse pair-full: 0.9300
- six-way full: 0.8494
- donor consistency: 0.9081 / 0.8531 / 0.8750

Causal controls remained clean:
- executive off: 0 on all rules
- canonical zero: ~0
- RULE_NONE: exact no-op
- admitted brain/base: bit-exact

Earlier representation diagnostics established 100% TRAIN→DEV cue decoding in
raw memory64, canonical recall128, and hidden state.

Therefore memory content is not the bottleneck.

## New diagnosis

The frozen motor acts on:

`historical_workspace + canonical_recall + executive_delta`

But v0.15's executive generated `executive_delta` from only:

`canonical_recall + rule`

It had no knowledge of the varying historical workspace baseline it was trying
to steer.

## v0.16.0 architecture

No memory changes.

Executive inputs:

- H = historical workspace128
- D = canonical recall128
- R = late rule

No raw memory64.
No raw hidden.
No image.
No sensor/context vector.

Features:

`C = tanh(Wc D)`
`X = tanh(Wx H)`
`R = tanh(rule_embedding)`

`[C, C*R, C*X, C*R*X]`
`→ Linear512→256`
`→ tanh`
`→ zero-init Linear256→128`
`→ workspace delta`

Every path contains C. Therefore D=0 forces the executive output to exact zero,
even though historical workspace and rule remain present.

This is contextual steering without allowing context to replace recalled content.

## Training change: modular recall is now learned explicitly

Every v3 TRAIN condition contains:

- six ordinary cue×rule objectives;
- six coherent donor-response objectives.

For the donor objective:
- recipient historical workspace is held fixed;
- canonical recall is recomputed from opposite-cue donor memory;
- target is the donor-cue action under the same rule.

Total:
`0.5 own objective + 0.5 coherent donor objective`.

Donor consistency is therefore now an intended learned operation, not an
emergent property.

Fresh v4—not v3—will later test whether it generalizes.

## TRAIN-only gate

v3 DEV is already spent and is not consulted.

Stages:
- update 1600 at lr 2e-4
- update 3200 at lr 1e-4 if needed
- update 4800 at lr 5e-5 if needed

Stop at first PASS.

Required overall:
- each rule pair-full >= 0.95
- each rule donor consistency >= 0.95
- executive-off <= 0.05
- canonical-zero <= 0.05
- six-way full >= 0.90
- RULE_NONE exact no-op

Every structural factor:
- each rule pair-full >= 0.85
- each rule donor >= 0.85
- six-way >= 0.75

## After PASS

Create a brand-new v4 family and evaluate the frozen model once.

If fresh v4 passes:
- single-item executive composition is complete;
- freeze/promote the contextual canonical executive;
- move immediately to multi-item/order-sensitive memory and model-based
  planning.

If v0.16.0 still cannot pass TRAIN by 4,800 updates:
- stop rule-mapping variants;
- redesign executive action control at a more fundamental level.

v3 TEST remains sealed.
