# Continuous Brain — Working Memory DEV Admission Gate v1

Status: **thresholds frozen before DEV evaluation**

This file defines the first held-out behavioral/causal admission gate for the
qualified 64-D writable working-memory mechanism.

The evaluator may use only DEV structural conditions `1600..1979`.
The TEST split `1980..2359` remains sealed.

## Independent unit

The independent experimental unit is `condition_id` (N=380), not the LEFT or
RIGHT episode individually.

A normal pair is fully correct only when the five-action argmax is the correct
LEFT action on the LEFT member **and** the correct RIGHT action on the RIGHT
member.

Side-only accuracy is diagnostic and never substitutes for full-action
competence.

## Pre-registered competence thresholds

Normal memory must satisfy all:

- pair full-action accuracy >= **0.70**;
- Wilson 95% lower bound for pair full accuracy >= **0.65**;
- LEFT member full-action accuracy >= **0.80**;
- RIGHT member full-action accuracy >= **0.80**;
- overall absolute LEFT/RIGHT member-accuracy gap <= **0.10**.

## Added-value requirement

The historical recurrent system is evaluated with working memory disabled from
step 1.

Normal-memory pair full accuracy must exceed this no-memory baseline by at least
**0.10 absolute**, and the paired one-sided exact discordance test must have
`p < 0.01`.

This prevents a causally active but behaviorally redundant memory module from
being admitted.

## Memory-content sufficiency

At the delayed decision, recurrent hidden state is reset to the initial zero
state while learned memory is retained.

The LEFT/RIGHT pair side-choice accuracy under this hidden-reset/own-memory
condition must be >= **0.65**.

This asks whether the learned 64-D state itself carries recoverable cue
information rather than merely shaping recurrent hidden state earlier.

## Causal memory swap

For every condition define:

`d = right_logit - left_logit`.

At the same matched delayed-decision state:

- LEFT recipient receives the RIGHT member's learned memory;
- RIGHT recipient receives the LEFT member's learned memory.

Donor-direction shifts are:

- LEFT recipient: `d_swap - d_normal`;
- RIGHT recipient: `d_normal - d_swap`.

Condition swap effect is their mean.

Admission requires all:

- mean condition effect > 0;
- deterministic 20,000-draw bootstrap 95% lower bound > 0;
- at least **70%** of conditions have positive swap effect;
- exact one-sided sign-test `p < 0.001`.

With recurrent hidden reset, swapped memory must also make both members prefer
the *donor* cue side in at least **65%** of condition pairs.

This is the primary causal criterion.

## Factor-sliced anti-collapse gate

Normal-memory performance is reported separately for every level of:

- corridor length;
- cue-visible duration;
- movement step/dynamics;
- horizontal goal offset;
- vertical goal offset;
- start-x translation;
- start-y translation;
- distractor pattern.

Every factor level must satisfy:

- pair full accuracy >= **0.50**;
- pair full accuracy no more than **0.25** below the overall DEV pair accuracy;
- absolute LEFT/RIGHT member-accuracy gap <= **0.25**.

One weak slice fails this DEV admission gate even when the aggregate passes.

## Additional controls

The evaluator records:

- memory disabled from step 1;
- memory forced zero from cue disappearance onward;
- memory zeroed only at the decision;
- normal memory;
- swapped memory;
- hidden reset + own memory;
- hidden reset + swapped memory;
- both hidden and memory reset.

For the both-reset condition, LEFT/RIGHT current sensory inputs are identical,
so logits must agree to maximum absolute difference <= `1e-6`, and pair
side-choice accuracy must be exactly zero.  This is an evaluator validity
check, not evidence for the mechanism.

## TEST rule

A DEV pass does not itself establish canonical admission.

Only after:
1. DEV evaluation code is reviewed;
2. DEV gate passes without threshold changes;
3. architecture/checkpoint/intervention definitions are frozen;

may the sealed TEST split be opened once with a separately packaged evaluator.
