# Continuous Brain v0.18.0 — Option-Intent Executive

## Result that motivated this design

v0.17.0 learned coherent donor action intent almost perfectly at update 4,800:
- invert_lr donor 0.9950
- vertical donor 0.9944
- vertical_reverse donor 0.9988

But ordinary own-content response control remained:
- 0.8981 / 0.9219 / 0.9319
- six-way 0.8900

The remembered content and donor semantics are therefore no longer the main
bottleneck.

## WHAT versus WHEN

v0.17 used one residual head both to preserve corridor motion after an
instruction appeared and to switch to the remembered/rule-conditioned response
at the decision boundary.

v0.18 separates these functions.

WHAT:
canonical recall + late rule -> raw action intent5

WHEN:
historical workspace + frozen base logits -> scalar execution gate

Final:
base logits + gate * raw intent

The gate never sees raw memory, canonical recall, or rule identity. It decides
when executive intent may act, not what remembered action is correct.

The WHAT path remains strictly recall-dependent:
- zero canonical recall -> exact zero action intent
- RULE_NONE -> exact zero action intent

## Training

v3 TRAIN only.

Per condition:
- six own RESPONSE losses
- six coherent donor RESPONSE losses
- gate BCE over post-instruction states:
  - pre-response gate 0
  - response gate 1

No primitive-action CE is imposed on intermediate post-instruction states.

Objective:
- 0.55 own response
- 0.25 donor response
- 0.20 timing gate

Stages:
- 1600 updates @ 2e-4
- 3200 @ 1e-4 if needed
- 4800 @ 5e-5 if needed

## Frozen TRAIN gate

- each rule pair-full >= 0.95
- each rule donor >= 0.95
- six-way >= 0.90
- executive off <= 0.05
- zero canonical content <= 0.05
- pre-response action preservation >= 0.99
- pre-response gate-off rate >= 0.99
- response gate-on rate >= 0.99
- RULE_NONE exact no-op
- strong factor slices

## After PASS

Create a brand-new v4 family and evaluate once.

A fresh-v4 PASS closes the single-item executive milestone. Then move to:
multi-item/order-sensitive memory, learned option sequences, model-based
planning, metacognitive confidence, and long-horizon 3-D embodiment.

If v0.18 cannot pass TRAIN by update 4,800, stop this primitive-action assay
family and move directly to temporally extended learned options.
