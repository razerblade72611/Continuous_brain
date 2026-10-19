# Continuous Brain v0.32.1 — Structurally Neutral Autobiographical Discrepancy Repair

## Source status

v0.32.0 remains a formal DEV FAIL.

It passed:
- reliability MAE: 0.02061
- preference accuracy: 1.00
- normal-history commitment: 1.00
- donor-history commitment flip: 1.00
- empty-history neutral: 1.00
- persistence: 1.00

Exactly one gate failed:

`expected_actual_discrepancy_load_bearing`

The expected==actual ablation did not become neutral.

## Diagnosis

The v0.32.0 model received:
- A/B primary label
- discrepancy features

as ordinary concatenated input to a biased encoder and GRU.

The ablation preserved A/B labels and chronology while setting all discrepancy
fields to zero.

That zero-discrepancy history is out-of-distribution relative to training.
Nothing in the original architecture forced the learned network to treat it as
neutral.

Therefore a learned default preference based on label/order structure could
remain even when discrepancy evidence disappeared.

This means the failed control did not distinguish:
- "discrepancy was unnecessary"
from
- "the architecture extrapolated non-neutrally when all evidence was removed."

v0.32.1 fixes the ambiguity structurally.

## Repair architecture

A/B labels are no longer standalone model evidence.

For each record:

`routed_A = primary_A * discrepancy_features`
`routed_B = primary_B * discrepancy_features`

The model sees only:

`[routed_A, routed_B]`

Thus A/B labels can determine which commitment a discrepancy belongs to, but a
label cannot generate evidence without discrepancy.

All learned evidence layers are bias-free:
- record encoder
- recurrent GRU
- output head

Output:

`reliability = 0.5 + 0.5 * tanh(evidence)`

Therefore:

zero discrepancy
-> routed evidence = 0
-> encoded evidence = 0
-> recurrent hidden = 0
-> output evidence = 0
-> reliability = exactly [0.5, 0.5]

This is true for:
- empty history
- one episode
- twelve episodes
- any A/B label sequence
- any history length

## Gradient sanity

Exact neutrality must not be produced by dead clamping.

At zero discrepancy:
- output must be exactly [0.5, 0.5]
- gradient from A-vs-B reliability difference to discrepancy channels must be
  finite and nonzero

So the neutral state remains learnable.

## Data use

TRAIN:
- original v0.32 TRAIN subjects 0..1023

DEV:
- original already-spent DEV subjects 3000..3255

TEST:
- subjects 6000..6255 remain sealed

No new DEV window is consumed.

The original v0.32 source model remains unchanged and its FAIL is preserved.

## Repair gates

Normal performance:
- reliability MAE <= .10
- preference accuracy >= .95
- normal history forms correct commitment >= .95
- donor history flips commitment >= .95

Structural neutrality:
- empty history -> exact [0.5,0.5] = 1.00
- expected==actual history -> exact [0.5,0.5] >= .95
- expected==actual bit-exact neutral = 1.00
- A/B labels + chronology with zero discrepancy -> exact neutral = 1.00
- standalone zero-discrepancy structural probe -> exact neutral
- zero-discrepancy gradient remains alive

Persistence:
- outputs bit-exact after model reload

## Meaning of PASS

A repair PASS supports the intended DEV claim much more cleanly:

Autobiographical A/B labels and chronology are insufficient by construction to
produce a commitment preference. Only expected-self versus actual-self
discrepancy can generate autobiographical evidence.

Normal discrepancy histories form the correct later commitment.
Donor histories transfer that later commitment.
Removing discrepancy while preserving labels, order, and history length forces
the self-model to exact neutrality.

The original v0.32.0 result remains FAIL. v0.32.1 is recorded separately as a
DEV repair.

This remains a functional computational result and does not establish
subjective experience, consciousness, sentience, qualia, or moral status.

## Stop rule

If PASS:
- freeze the v0.32.1 structurally neutral autobiographical model
- run one frozen TEST on subjects 6000..6255
- never tune against that TEST afterward

If FAIL:
- do not touch TEST
- stop adding readout capacity
- inspect whether the expected/actual trajectory records themselves contain
  enough commitment-specific evidence on DEV
