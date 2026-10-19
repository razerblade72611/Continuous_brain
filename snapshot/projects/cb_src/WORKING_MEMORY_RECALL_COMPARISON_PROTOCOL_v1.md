# Continuous Brain — Working Memory Recall Comparison v1

Status: **pre-training qualification design; DEV/TEST remain sealed**

## Why this experiment exists

Working Memory M1 demonstrated:

- a real differentiable 64-D writer;
- exact historical-base preservation;
- short-delay bidirectional competence;
- severe long-delay LEFT degradation;
- and, crucially, a length-dependent RIGHT-minus-LEFT pull in the frozen 14k
  substrate even after cue erasure, recurrent-hidden reset, and external memory
  forced to zero.

M1 therefore showed that a learned writer feeding only the inherited reader is
not sufficient for robust long-delay behavior.

The next hypothesis is not simply "add a smarter recall module."  We must first
separate the benefit of **direct workspace access** from any benefit of
context-sensitive learned recall.

## Arms

### M1 — writer only, inherited frozen reader

Already completed.  This is the negative reference specimen.

The learned memory reaches behavior through:

    m_t
      -> frozen historical memory projection
      -> frozen recurrent core
      -> frozen workspace
      -> frozen consumers

M1 is not retrained for this qualification.  Its existing result is preserved.

### M1.5 — direct linear recall control

Adds only:

    r_t = P_0(m_t)
    workspace'_t = workspace_t + r_t

where `P_0` is a bias-free memory->workspace Linear with its weight initialized
exactly to zero.

There is:
- no gate;
- no bias;
- no memory->motor shortcut;
- no hidden->workspace trainable shortcut.

This is the simplest test of whether shortening the memory path and permitting
learned gain is sufficient.

### M2 — gated contextual recall

Uses the **same** zero-initialized projection:

    decoded_t = P_0(m_t)

and adds only a contextual multiplicative gate:

    g_t = sigmoid(G([h_t, m_t]))
    r_t = g_t * decoded_t
    workspace'_t = workspace_t + r_t

`G` starts at zero, so `g_t = 0.5` initially.

The gate cannot act as an independent hidden->workspace adapter because it only
multiplies decoded memory.  If `m_t` carries no content, the direct recall
residual is zero.

## Exact-preservation rule

M1.5 and M2 must both be bit-exact no-ops at construction.

The direct memory projection is bias-free and exactly zero initialized.
Therefore the direct workspace residual is exactly zero for **arbitrary**
memory input before learning.

The protected 14k historical brain must remain bit-exact after every optimizer
step.

## Staged-gradient qualification

The writer itself remains the already-qualified M1 writer and the historical
memory route remains active.  This creates a useful staged prediction.

### Update 1

Memory starts exactly zero.

Required:
- writer candidate gradient > 0;
- direct recall projection gradient exactly 0;
- M2 gate gradients exactly 0.

This proves the new direct path cannot learn a constant task correction before
memory contains information.

### Update 2

The writer has moved and memory becomes nonzero.

Required:
- writer remains gradient-bearing;
- direct recall projection gradient > 0;
- M2 gate gradients remain exactly 0 during this forward because the projection
  is still zero before the update.

### Update 3

The direct projection has now moved.

Required:
- recall residual becomes nonzero;
- direct projection remains gradient-bearing;
- in M2, gate weight and bias gradients become > 0.

This demonstrates that the gate becomes load-bearing only after decoded memory
can actually affect the workspace.

## Qualification gates

Both M1.5 and M2 must pass independently in fresh Python processes:

1. canonical checkpoint SHA exact;
2. historical base digest exact;
3. complete LEFT/RIGHT initial no-op trajectory bit-exact;
4. delayed recurrent writer graph still enters at step 2;
5. staged writer/projection/gate gradients follow the predictions above;
6. learned direct recall residual becomes nonzero;
7. historical base remains bit-exact after every update;
8. no reported DirectML CPU fallback.

Failure of either arm stops the comparison before training.

## Interpretation matrix — frozen before training

### M1.5 ~= M2, both materially better than M1

Conclusion:
**directness / workspace access / gain is sufficient**.

Do not claim a cognitively special role for gating.

Prefer the simpler M1.5 architecture unless another task later demonstrates a
need for context-sensitive recall.

### M1.5 partially improves, M2 improves further

Conclusion:
**richer contextual recall is promising**, but this does *not yet* prove that
gating specifically is responsible, because M2 has more trainable parameters.

Before attributing the gain specifically to gating, run a parameter-matched
ungated control.

### M1.5 and M2 both fail similarly to M1

Conclusion:
direct workspace readout is not sufficient.

Return to writer/storage stability and recurrent integration rather than adding
more reader complexity.

### M1.5 succeeds and M2 is worse

Conclusion:
the contextual gate is harmful or unnecessary under this task family.

Prefer M1.5 and diagnose the gate before any further complexity.

## Behavioral reporting requirement

Future train-side and DEV evaluators must report **two separate quantities**:

1. full-action competence / lateral-action-top-choice rate;
2. LEFT-vs-RIGHT side-sign correctness among the lateral logits.

The cue-erased frozen-substrate probe showed that the full argmax can collapse
to `down` while a strong RIGHT-over-LEFT latent preference persists.  A single
accuracy number would conflate these two failures.

## Split discipline

- qualification uses TRAIN-family conditions only;
- production smoke uses TRAIN only;
- full arm training uses TRAIN only;
- DEV remains unopened until an arm clears train-side mechanistic gates;
- TEST remains sealed until architecture, evaluator, and thresholds are frozen.

## Lineage rule

M1.5 and M2 must each start from the untouched canonical 14k checkpoint, never
from the trained M1 checkpoint and never from each other.

M1 remains preserved as the writer-only/frozen-reader negative reference.
