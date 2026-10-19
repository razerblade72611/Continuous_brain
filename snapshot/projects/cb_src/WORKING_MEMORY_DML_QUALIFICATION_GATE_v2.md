# Continuous Brain — Working Memory DirectML Qualification Gate v2

Status: **pre-training mechanism qualification, not canonical admission**

## Locked scientific claim

This gate asks only whether an additive writable working-memory mechanism can be
connected to the protected 14k InfantBrain, trained through full BPTT on
DirectML, and updated without mutating any historical base tensor.

It does **not** claim that the memory is useful, general, conscious, sentient, or
architecturally admitted.

## Protected base

Canonical historical checkpoint:

`state/child_v113_dml_smoke/infant_training.pt`

Required SHA-256:

`086932388a9a410af46fea8e87cc63dd4b60161e03e247bf81fc88d28cfd577f`

All base tensors and buffers are compared bit-for-bit before and after optimizer
updates.

The previously qualified DirectML compatibility path remains in force:

- historical base parameters are optimizer-frozen;
- only `base.nucleus.workspace` has `requires_grad=True` for DirectML autograd
  compatibility;
- workspace parameters are explicitly excluded from optimizer ownership;
- only the new working-memory writer is optimizer-owned;
- workspace gradients are cleared at every optimizer boundary;
- base digest must be identical afterward.

## Working-memory mechanism

The additive writer is:

    candidate_t = tanh(W_c h_t)
    gate_t      = sigmoid(W_g [h_t, m_t])
    m_(t+1)     = m_t + gate_t * (candidate_t - m_t)

The protected InfantBrain receives `m_t` through its already-existing projected
memory stream on the next recurrent step.

`candidate` and `write_gate` are both zero-initialized. With `m_0 = 0`,
candidate output is exactly zero at construction, so memory remains exactly
zero and the wrapper is an exact behavioral no-op before learning.

## Counterfactual-balanced optimizer rule

No optimizer update may be based on a single randomly selected cue.

Each update is:

    one structural condition_id
             |
       +-----+-----+
       |           |
      LEFT       RIGHT
       |           |
    full BPTT   full BPTT
       |           |
       +-----+-----+
             |
         mean loss
             |
       ONE optimizer step

This structurally prevents cue-frequency or update-order imbalance from
producing a directional gradient on a nominally balanced update.

Qualification uses only train-family condition IDs and never accesses dev/test
conditions.

## Eight required qualification gates

All eight must pass before long memory training:

1. **Exact initialization no-op**  
   On both LEFT and RIGHT members of the same varied condition, memory-enabled
   execution must be bit-identical to the exact base path for hidden state,
   workspace, and motor logits over a complete teacher trajectory.

2. **Causal base-memory path**  
   At a cue-hidden state, changing only the memory input must change recurrent
   hidden state and policy logits. Opposite memory probes must also produce
   different logits.

3. **Fixed memory shape**  
   Memory must remain exactly `[1,64]` throughout all trajectories.

4. **Candidate gradient through full BPTT**  
   On the first exactly balanced LEFT/RIGHT update, task loss must give a
   finite, nonzero gradient to the zero-initialized candidate projection.

5. **Write-gate gradient after candidate moves**  
   After the candidate receives one optimizer update and memory becomes
   nonzero, the next exactly balanced pair loss must give a finite, nonzero
   gradient to the write gate.

6. **Nonzero learned memory**  
   After update 1, update-2 trajectories must contain nonzero memory state.

7. **Historical-base bit-exact preservation**  
   Every base parameter and buffer must remain exactly unchanged after both
   optimizer steps, and the whole-base digest must match.

8. **No reported DirectML CPU fallback**  
   The captured qualification log must contain no DirectML CPU fallback
   warning.

Failure of any gate means **stop; do not begin long training**.

## Pre-registered later admission criterion

Passing this DirectML qualification does not admit working memory.

After qualification, training will use the already-admitted varied
counterfactual task family with frozen train/dev/test structural splits.

Before seeing model results, the behavioral admission criteria are fixed:

### Pair-level success

The independent statistical/behavioral unit is `condition_id`, not episode.

A condition is solved only when its LEFT and RIGHT counterfactual members both
satisfy the required behavioral criterion.

### Causal memory intervention

For the same held-out condition:

1. run the LEFT member and capture learned memory after the cue is hidden;
2. run the RIGHT member and capture learned memory at the matched point;
3. continue each condition normally;
4. repeat while swapping LEFT memory into RIGHT and RIGHT memory into LEFT;
5. also evaluate memory forced to zero;
6. reset recurrent hidden state while preserving memory;
7. reset both memory and recurrent hidden state.

Admission requires more than an accuracy correlation:

> Swapping recalled content must predictably alter the later cue-dependent
> decision in the direction implied by the swapped content.

If memory can be zeroed/swapped without the predicted behavioral change, the
mechanism has not demonstrated causal use and fails admission.

### Factor-sliced held-out reporting

Before any aggregate held-out result is accepted, report pair-level performance
separately by every generator factor:

- corridor length;
- cue-visible duration;
- movement step/dynamics;
- horizontal goal offset;
- vertical goal offset;
- start-x translation;
- start-y translation;
- distractor pattern.

This is mandatory even if the aggregate number looks excellent. A collapse on
one factor slice cannot be hidden inside a high average.

### No test-set tuning

Development decisions use only train and dev structural conditions.

The frozen test range (`1980..2359`) is opened only after architecture,
interventions, thresholds, and factor-sliced reporting are fixed.
