# Continuous Brain v0.19.0 — Recurrent Option Progress Controller

## Result that motivates this branch

v0.18.0 established that WHAT is essentially solved on v3 TRAIN:

- invert_lr pair-full 0.9912
- vertical pair-full 0.9906
- vertical_reverse pair-full 0.9931
- six-way 0.9862
- donor consistency 0.9988 / 0.9938 / 0.9906

But WHEN failed:

- pre-response preservation 0.5630
- gate-off 0.5550
- gate-on 0.8984.

v0.18.1 then tested whether response timing is statically observable from
candidate protected states. No candidate reached the frozen 0.95 observability
threshold.

Best static protected neural representations reached only roughly 0.74–0.75
boundary accuracy.

Therefore the response boundary must not be treated as another static
classification problem.

## Architectural decision

Freeze v0.18 WHAT.

Do not change:
- memory writer;
- canonical recall;
- historical brain;
- world model;
- motor;
- learned v0.18 content+rule -> raw action intent mapping.

Add one new recurrent option-progress state for WHEN.

Protected inputs to WHEN:
- protected post-core hidden;
- historical workspace;
- frozen base motor logits;
- previous option-progress state.

The progress controller does NOT receive:
- raw memory64;
- rule identity;
- image;
- raw observation/context.

The rule is used only as an active/reset signal:
- RULE_NONE -> progress state = exactly zero, gate = exactly zero;
- active late instruction -> recurrent state advances.

## Recurrent execution gate

`hidden -> 128`
`workspace -> 64`
`base logits -> 32`

concatenate
`-> input fusion 224->128`

then

`option_state(t+1) = tanh(input_t + W_rec option_state(t))`

and

`execution_gate = 2 * (sigmoid(W_gate option_state) - 0.5)`.

The gate output layer is zero initialized, so growth is an exact no-op.

## Parent WHAT preservation

The source is exactly:

`state/option_intent_executive_v1/latest.portable.pt`

and must be the v0.18 update-4800 checkpoint.

Its entire executive state is loaded and frozen.

Only:
- recurrent progress controller;
- scalar intent_gain

are trainable.

This makes the experiment attributable to temporal state rather than another
simultaneous rewrite of recalled content or action intent.

## DML-safe timing objective

v0.18 used BCEWithLogits, which triggered a DirectML CPU fallback.

v0.19 uses no BCE.

Timing uses direct MSE on the execution gate:
- pre-response target 0;
- response target 1.

The runner treats any reported CPU fallback as failure.

## TRAIN objective

Per v3 TRAIN condition:

- timing MSE: 0.45
- pre-response action preservation: 0.20
- own response: 0.20
- coherent donor response: 0.10
- left/right timing invariance: 0.05

BPTT is retained across the entire active option phase.

The cue-invariance term prevents the scalar WHEN controller from becoming a
hidden substitute for the WHAT pathway.

## Frozen TRAIN admission

Overall:
- each rule pair-full >= 0.95
- each donor pair >= 0.95
- six-way >= 0.90
- executive off <= 0.05
- zero recalled content <= 0.05
- pre-response action preservation >= 0.99
- pre-response |gate|<0.5 rate >= 0.99
- response gate>=0.5 rate >= 0.99
- mean absolute pre gate <= 0.10
- mean response gate >= 0.90
- left/right cue gate-gap mean <= 0.05
- cue gate-gap p99 <= 0.15
- RULE_NONE exact no-op

Factor levels:
- each rule pair >= 0.85
- each donor >= 0.85
- six-way >= 0.75
- pre preservation >= 0.95
- gate-off/on >= 0.95
- cue gate-gap mean <= 0.10

## Held-out policy

v3 DEV remains scientifically spent and is not used.
v3 TEST remains sealed.

If TRAIN passes:
- create a brand-new v4 family;
- freeze v0.19;
- evaluate v4 once.

If fresh v4 passes:
- close the single-item executive milestone;
- move to multi-item/order-sensitive memory and model-based planning.

If v0.19 cannot pass TRAIN by update 4800:
- stop this single-item assay family;
- do not build another gate variant;
- move to a more general learned option/state machine in the planning stack.
