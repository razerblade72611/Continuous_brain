# Continuous Brain v0.15.0 — Rich Canonical Executive

## What the v0.14.1 convergence run proved

Training the original 16,896-parameter canonical executive to 8,000 updates did
not solve TRAIN. At update 8,000:

- invert_lr pair-full ≈ 0.685
- vertical pair-full ≈ 0.849
- vertical_reverse pair-full ≈ 0.863
- six-way full ≈ 0.676

All executive-off and canonical-zero controls still collapsed, and the admitted
brain remained bit-exact.

Earlier representation diagnostics showed:

- raw memory64 cue decoding TRAIN→DEV = 1.000
- canonical recall128 cue decoding TRAIN→DEV = 1.000
- recurrent hidden cue decoding TRAIN→DEV = 1.000

Therefore:
- memory content is present;
- canonical recall content is present;
- more optimization of the old transform is insufficient;
- the old executive transform is the bottleneck.

## Architectural change

Memory architecture: unchanged.

Old executive:
`W_out(tanh(content) * tanh(rule))`

New rich executive:

`content128 -> Linear128->128 -> tanh`

`rule -> embedding128 -> tanh`

`concat(content, rule, content*rule)`
`-> Linear384->256 -> tanh`
`-> zero-init Linear256->128`
`-> workspace delta`

This adds a learnable content rotation and richer content/rule interaction while
retaining strict causal isolation:

- no raw memory64 read;
- no direct hidden/image/context input;
- same canonical recall feeds admitted workspace and executive;
- RULE_NONE is exact no-op.

Trainable parameters: approximately 148k, still tiny relative to the protected
brain.

## TRAIN-only architecture gate

v3 DEV is already scientifically spent and MUST NOT be used to select this
architecture.

Train only on v3 TRAIN.

Stages:
- epoch 1 / update 1600: lr 2e-4
- epoch 2 / update 3200: lr 1e-4 if needed
- epoch 3 / update 4800: lr 5e-5 if needed

Stop at first fixed-final TRAIN PASS.

Frozen requirements:

For each rule:
- pair full >= 0.95
- coherent canonical donor >= 0.95
- executive-off <= 0.05
- canonical-zero <= 0.05

Across all six:
- six-way full >= 0.90

RULE_NONE:
- exact no-op <= 1e-6

Every structural factor:
- each rule >= 0.85
- six-way >= 0.75

## After PASS

Do not evaluate on v3 DEV.

Create a brand-new v4 structural family with a new held-out DEV/TEST split and
evaluate the frozen rich executive exactly once.

If v4 generalization passes, the single-item executive milestone is complete and
the project moves to multi-item/order-sensitive memory and model-based planning.

If this richer transform cannot pass TRAIN by update 4800, stop adding training
time; the executive formulation needs a deeper redesign.
