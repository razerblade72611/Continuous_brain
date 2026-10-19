# Working Memory Parameter-Matched Ungated Recall Control v1

## Purpose

M2 has 49,280 recall parameters; M1.5 has only 8,192.

The final fixed TRAIN advantage of M2 over M1.5 is small enough that it cannot
be credited to contextual gating until capacity/nonlinearity is controlled.

This control matches M2's recall parameter count **exactly** while removing all
contextual hidden-state input and all contextual gating.

## Architecture

    m[64]
      -> bias-free Linear 64->256
      -> tanh
      -> bias-free Linear 256->128, zero initialized
      -> static learned per-workspace scale[128], initialized to 1
      -> additive workspace residual

Trainable recall parameters:

- 64*256 = 16,384
- 256*128 = 32,768
- scale 128 = 128

Total recall = **49,280**, exactly equal to M2.

Writer = 36,992.

Total additions = **86,272**, exactly equal to M2.

## Anti-shortcut constraints

- no hidden/context input;
- no bias terms;
- zero-initialized output layer;
- exact behavioral no-op at construction for arbitrary memory;
- static scale cannot create output when decoded memory is zero.

## Staged qualification

Update 1:
- writer candidate grad >0;
- output grad =0;
- feature transform grad =0;
- static scale grad =0.

Update 2:
- memory nonzero;
- output grad >0;
- feature transform grad =0;
- static scale grad =0.

Update 3:
- recall residual nonzero;
- output grad >0;
- feature transform grad >0;
- static scale grad >0.

Historical base must remain bit-exact and DirectML must report no CPU fallback.

## Interpretation after matched full training

If this control matches M2:
- M2's small advantage is explained by capacity/nonlinearity;
- contextual gating is not established as necessary.

If M2 materially exceeds this control:
- stronger evidence that using current hidden/context to modulate recall adds
  value beyond matched memory-only capacity.

If the control exceeds M2:
- contextual gating is unnecessary or harmful for this task family.

DEV and TEST remain sealed until this control is complete.
