# Persistent Individual Core — implemented architecture

## Two levels, not a replacement motor policy

The executive chooses temporally extended options. A frozen imported v0.5.x
GRU executes motor control; without an import, the development room uses an
explicit analytical reference controller. No legacy learned weight is modified.

Perception, the empirical self-state, an intention embedding and retrieved
episodic records become tokens. Four recurrent workspace slots attend to these
tokens through Transformer decoder blocks, then pass through a shared GRU update.
Their normalized pooled state produces option values. A separate action-conditioned
GRU predicts future feature vectors, reward, task success and observation variance.

The standard configuration is width 768, three attention blocks, eight heads,
four workspace slots, six options, four initial goal types. No assertion is made
that this size is sufficient for consciousness or general intelligence.

## Skill representation and composition

Initial options: observe, orient(target), navigate(target), inspect(target),
search(region=current_view), rest. Each records preconditions, an expected effect,
controller, bounded duration and named failure modes. Empirical success probability
uses a Beta(1,1) posterior mean; mean cost and failure counts accumulate for life.
These are statistical estimates, not a learned causal theory of self.

A generic backward-chainer can resolve `inspected` through `near`, `aligned` and
`visible`, skipping already satisfied requirements. It supplies a teacher label,
not a hidden door-task routine. Only currently legal options enter selection.
The initial scheduler selects inspect/reach, occasional exploration and bounded
energy restoration. Goal invention and skill discovery are future research work.

## Learning

Each replay row contains the raw cognitive packets before and after an option,
the initial workspace, masks, option, teacher label, discounted reward, duration,
task outcome, and a physical/goal boundary marker. The default sequence length
is six decisions, with two burn-in decisions and gradients over the remaining four.
Long-lived hidden state does not imply long-horizon backpropagation: this is
explicitly truncated sequence learning, with longer horizons left configurable
in code for later experiments.

Loss = option TD + 0.15 planner imitation + predictive losses. Predictive losses
include feature Gaussian likelihood, reward Huber loss, outcome BCE, and a small
one-step latent consistency loss. Rollouts extend up to three option transitions
and stop at physical/goal boundaries. Target weights use exponential averaging;
gradients are clipped and checked for finiteness. Negative Gaussian loss values
are possible as predicted variance shrinks; they are not automatically errors.

Planner forcing decays from 1.0 toward 0.10 over 5,000 decisions; non-teacher choices
have 10% exploration. After 200 updates a small uncertainty-penalized predicted
consequence term reranks option values. This is a preliminary learned consequence
model, not the deliberative world simulator intended for v0.8.

The replay window is bounded at 2,048 options. Sampling balances represented
goal groups, with 20% formative replay from up to 32 early contiguous sequences.
This mitigates forgetting; it does not prove lifelong retention of arbitrary skills.
Offline consolidation repeats the same objective. Generative replay is not implemented.

## Persistence and fault handling

The authority is a single SQLite checkpoint blob plus typed memory rows committed
in the same transaction. The checkpoint includes online/target model, optimizer,
workspace, replay, protected sequences, self-model, intention/history, environment,
RNG and frozen motor state. The byte digest is validated on load. Dimension or
environment mismatches fail rather than recreating an identity.

Memory queries run every decision and return stable sensory/outcome records to
attention. Candidate retrieval mixes a recent window with a formative window.
It never queries diagnostic rows. All records remain available in the database.

A healthy option commits its proposed workspace only after receiving its outcome.
If any primitive has a sensor/infrastructure fault, the entire option is excluded
from replay, reward and skill-success accounting. The fault is recorded separately.
Recovery may reacquire the physical scene without zeroing the executive workspace.
Physical scene changes invalidate/re-ground target-level intentions as needed;
lifetime history survives them. Learned latent state is not proof of sentience.

## Next research milestones

1. Multi-seed unassisted executive evaluation, delayed-information tasks and
   ablations removing workspace recurrence/memory retrieval.
2. Longer sequence training, learned termination/options, richer target parameters
   and explicit retention measurements. Expand dimensions only with a tested migration.
3. Source-aware semantic VLM/LLM cognition, semantic/social memory and language goals.
4. Longer-horizon world modeling, calibrated uncertainty and stronger deliberation.
5. Cross-environment transfer and eventual independent physical safety supervision.

This release establishes trainable interfaces and persistence, not the completion
of every item in the proposed long-term Persistent Individual architecture.
