# Continuous Brain — Working Memory DirectML Qualification Gate v3

Status: **pre-training mechanism qualification, not canonical admission**

This is a narrow repair of v2. No architecture, checkpoint, optimizer, task
family, train/dev/test split, or scientific admission criterion changed.

## Why v2 stopped

v2 incorrectly required the recurrent hidden tensor to have a `grad_fn` on the
very first step.

For this additive memory architecture that is not the correct graph topology:

1. step 1 enters the frozen historical recurrent core with `m_0 = 0`;
2. therefore step-1 recurrent hidden is expected to be non-differentiable with
   respect to the new writer;
3. the trainable writer computes `m_1` *after* that recurrent step;
4. `m_1` must itself have a differentiable graph;
5. on step 2, `m_1` enters the historical memory projection and recurrent core;
6. therefore recurrent hidden must acquire a `grad_fn` beginning on step 2.

v3 now tests exactly that delayed graph-entry contract rather than importing the
step-1 rule from Competitive Broadcast, whose trainable residual entered the
recurrent computation immediately.

## DirectML diagnostic repair

v2 also used `torch.count_nonzero()` directly on a DirectML tensor for an
exact-zero assertion. PyTorch DirectML reported that diagnostic operator as a
CPU fallback.

v3 does not weaken the zero test. It performs an explicit `.detach().cpu()`
copy and then checks exact zero on CPU. Explicit device transfer for inspection
is not an operator fallback in the model computation.

Workspace gradient-boundary checks now use the stronger `grad is None`
assertion after `zero_grad(set_to_none=True)`, avoiding another potential
DirectML `count_nonzero` fallback.

## Gates unchanged

The mechanism still must pass all of these before long training:

1. exact LEFT/RIGHT enabled-path no-op at initialization;
2. causal memory-input effect on hidden state and logits at a cue-hidden state;
3. fixed `[1,64]` memory shape;
4. nonzero candidate-writer gradient through delayed full BPTT;
5. nonzero write-gate gradient after candidate weights move;
6. nonzero learned memory;
7. bit-exact historical InfantBrain preservation after optimizer steps;
8. no reported DirectML CPU fallback.

Every optimizer update remains an exact LEFT/RIGHT counterfactual pair from the
same structural `condition_id`.

## Pre-registered later admission criterion unchanged

Passing qualification does not admit working memory.

Held-out admission still requires pair-level performance, mandatory factor
slices, memory-zero and hidden-reset controls, and the decisive causal
intervention:

> swapping learned LEFT/RIGHT memory contents within the same held-out
> condition must predictably alter the later cue-dependent decision in the
> direction implied by the swapped memory.
