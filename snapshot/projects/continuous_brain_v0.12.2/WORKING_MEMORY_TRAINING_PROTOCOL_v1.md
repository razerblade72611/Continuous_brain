# Continuous Brain — Working Memory Training Protocol v1

Status: **pre-registered production training protocol; working memory is still experimental**

## What qualified before this protocol

The varied counterfactual environment passed its own independent admission gate.

The DirectML working-memory mechanism then passed all eight qualification gates:

- exact LEFT/RIGHT initialization no-op;
- causal base-memory path;
- fixed `[1,64]` memory state;
- nonzero candidate gradient through delayed temporal credit;
- write-gate gradient after candidate movement;
- nonzero learned memory;
- historical base bit-exact after optimizer updates;
- no reported DirectML CPU fallback.

This protocol does not reopen those gates. It uses the qualified mechanism.

## Intentional architectural isolation

For this milestone there is **no direct trainable recurrent residual**.

The only new plastic path is:

    recurrent hidden h_t
            |
            v
      64-D memory writer
            |
            v
          m_(t+1)
            |
            v
    existing frozen memory projection
            |
            v
    frozen recurrent core at t+1

Therefore step 1 has no writer-to-hidden gradient path; recurrent hidden acquires
trainable dependence beginning at step 2. This isolation is intentional so the
later causal intervention can be attributed to writable memory rather than a
second recurrent adapter.

A recurrent adapter may be tested later as a separate mechanism if the 64-D
memory bottleneck proves limiting.

## Frozen training split

Only condition IDs `0..1599` are available to training.

A deterministic permutation visits conditions **without replacement within each
1600-update epoch**. The schedule is a pure function of update number and
`schedule_seed=73191`, so resume cannot silently change examples.

Every update contains both counterfactual members of exactly one condition:

    same condition
       /      \
    LEFT      RIGHT
       \      /
       mean loss
          |
    one Adam update

Cue balance is therefore exact at every optimizer step, not merely in aggregate.

## Production objective

The previous Competitive Broadcast training exposed a dangerous objective
imbalance: many cue-independent locomotion actions can dominate one delayed
cue-dependent choice.

Working-memory training therefore pre-registers:

    motor_objective
      = 0.50 * mean_CE_over_all_steps
      + 0.50 * CE_at_first_delayed_side_decision

    total_loss
      = motor_objective
      + 0.25 * mean_world_prediction_loss

LEFT and RIGHT member objectives are then averaged.

The `0.50` decision fraction is fixed before the production run. Changing it
after seeing held-out results creates a new experiment.

## Preservation contract

- base checkpoint SHA must remain
  `086932388a9a410af46fea8e87cc63dd4b60161e03e247bf81fc88d28cfd577f`;
- historical base is optimizer-frozen;
- workspace receives DirectML compatibility gradients only;
- workspace is never optimizer-owned;
- workspace is checked exactly every update;
- entire historical base is checked exactly every 50 updates and at completion;
- disabling working memory after training must reproduce the exact zero-memory
  historical-base path;
- checkpoints contain only CPU tensors and must load with ordinary
  `torch.load(..., weights_only=True)`.

## Training phases

### Phase 0 — production-trainer smoke

Run the exact production trainer for 25 updates.

This is not a toy model or alternate algorithm. It is the production path at a
short horizon to qualify serialization, scheduling, paired loss, invariants,
and DirectML behavior.

Required:

- exit 0;
- no CPU fallback;
- nonzero candidate gradient each update;
- write-gate gradient after update 1;
- exact workspace every update;
- full base checks pass;
- final base digest unchanged;
- memory-disabled exact-base parity;
- portable checkpoint written.

No scientific performance claim is made from 25 updates.

### Phase 1 — first full train epoch

Only after smoke passes:

    updates = 1600

This visits every frozen train structural condition exactly once as a LEFT/RIGHT
pair.

No test-set conditions are touched.

## Pre-registered development analysis

After Phase 1, use only DEV conditions `1600..1979`.

The independent unit is `condition_id`. A pair succeeds only if both its LEFT
and RIGHT members meet the criterion.

Mandatory reporting, even if aggregate results are strong:

- overall pair-level delayed-decision accuracy;
- LEFT and RIGHT member accuracy separately;
- corridor length slices;
- cue-visible-duration slices;
- movement-step/dynamics slices;
- horizontal goal-offset slices;
- vertical goal-offset slices;
- start-x slices;
- start-y slices;
- distractor-pattern slices.

No single aggregate number can admit the mechanism.

## Pre-registered causal interventions

On the same held-out DEV conditions evaluate:

1. normal memory;
2. memory disabled/forced zero after cue disappearance;
3. LEFT/RIGHT memory snapshot swapped within the matched condition;
4. recurrent hidden reset while own memory is preserved;
5. hidden reset plus memory swap;
6. both hidden and memory reset.

Primary causal quantity for swap:

For each condition let `d = right_logit - left_logit` at the delayed decision.

Expected donor-direction shifts are:

- LEFT recipient receiving RIGHT memory: `d_swap - d_normal > 0`;
- RIGHT recipient receiving LEFT memory: `d_normal - d_swap > 0`.

The condition-level swap effect is the mean of those two shifts.

Working memory cannot be admitted merely because normal accuracy is high.
Admission requires the distribution of condition-level swap effects to be
reliably positive and the zero/reset controls to behave in the predicted
direction.

Exact numerical behavioral thresholds will be frozen in the evaluator before
the 1600-update model is evaluated; the sealed TEST split remains unopened until
all architecture and threshold decisions are complete.
