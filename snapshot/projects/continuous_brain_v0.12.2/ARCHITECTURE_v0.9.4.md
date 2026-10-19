# Continuous Brain v0.9.4 architecture

## Core thesis

The individual is the persistent child-owned system. Pretrained models are replaceable parents. Parent weights are never part of the child's checkpoint.

## Dual visual development

```text
raw pixels
   |
   +---------------------------+
   |                           |
   v                           v
VENTRAL / WHAT             DORSAL / WHERE-HOW
semantic visual cortex     spatial visual cortex
(child-owned)              (child-owned)
   |                           |
DINO global parents        simulator state now;
Qwen language lessons      V-JEPA/patch/depth parents later
   |                           |
   +-----------+---------------+
               v
        identity-preserving
             fusion
               |
               v
      recurrent cognitive nucleus
```

The fusion layer starts as `[I, 0]`: before dorsal learning contributes anything, the nucleus receives exactly the pre-v0.9.4 semantic visual bus.

## Predictive development

```text
slow EMA dorsal state(t)
          +
       action(t)
          +
       workspace(t)
          |
          v
  world-model ensemble
      /       |        \
     v        v         v
 spatial   structured  outcome
 delta      state      reward/success
     |        |         |
     +--------+---------+
              v
     predicted consequence
```

The slow EMA target is child-owned. It is not DINO, V-JEPA, or any other parent. It exists to stabilize the child's own predictive coordinate system while the online cortex remains plastic.

## Why parents remain useful

- DINO parents teach the ventral stream semantic visual invariances.
- Qwen parents teach grounded language, explanations, reasoning curricula, and corrections.
- V-JEPA / patch-feature parents can later teach the dorsal/temporal stream richer motion features.
- Whisper can parent the auditory cortex.
- Simulator state/depth/physics provides precise childhood teaching signals unavailable in the real world.

Parents can be upgraded. Parent lessons retain provenance. The child continues.

## World-model gate

A useful world model must do more than predict a plausible next state. It must prove action-conditioned causality:

```text
error(correct executed action) < error(wrong action)
```

and beat the honest no-change baseline in both dorsal latent space and structured state space.
