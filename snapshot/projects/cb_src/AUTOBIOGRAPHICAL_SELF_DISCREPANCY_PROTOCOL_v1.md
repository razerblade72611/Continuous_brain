# Continuous Brain v0.32.0 — Autobiographical Expected-Self / Actual-Self Discrepancy

## Starting point

v0.31.1 is a frozen TEST PASS.

On all 1024 sealed trajectories:
- teacher-forced world/self/intention metrics passed;
- eight-step autonomous world/self/intention metrics passed;
- explicit intention causally selected the matching imagined future;
- scalar-only reconstruction remained at chance;
- donor-primary intervention flipped the imagined future;
- surprise temporarily recruited INVESTIGATE without erasing primary intention;
- primary intention resumed when uncertainty resolved.

The v0.31 TEST is now spent and closed.

## New question

Can the continuing agent remember:

> "This is who I predicted I would become if I followed this course"

and later compare that with:

> "This is who I actually became"

then use the accumulated pattern of those discrepancies to change later
commitment formation?

v0.32 tests exactly that.

## Autobiographical record

Before acting, the frozen v0.31 model generates an expected four-step trajectory:

- expected future continuous self at every step;
- expected future world at every step;
- expected temporary cognitive mode at every step.

After the plan is executed, v0.32 stores:

- actual continuous self trajectory;
- actual world trajectory;
- actual temporary mode trajectory.

The autobiographical record therefore contains both the expectation and the
experience.

It is not a success/failure label.

## Record features

The history learner receives only features derived from the difference between
expected and actual trajectories:

- commitment identity A/B;
- mean absolute self discrepancy6;
- final absolute self discrepancy6;
- mean signed self discrepancy6;
- mean world discrepancy4;
- mode mismatch rate;
- total mean self discrepancy;
- terminal self discrepancy.

It does NOT receive:
- identity embedding;
- subject ID;
- current world;
- current self;
- current goal;
- hidden reliability profile.

This prevents the history learner from simply memorizing identities or reading
the answer from the present state.

## Development subjects

Each synthetic subject has a hidden, persistent action/self-consistency profile.

One commitment tends to be reliable:
- ~0.90 to 0.95

The other tends to be unreliable:
- ~0.05 to 0.15

Which commitment is reliable alternates independently by subject ID parity.

Every subject lives through 12 autobiographical episodes:
- 6 commitment-A episodes
- 6 commitment-B episodes

The frozen v0.31 model forms each expected trajectory before the episode.

Actual episodes can contain exogenous failures according to the subject's hidden
reliability profile.

The history learner must infer future self-reliability from the discrepancy
history alone.

## Model

Each autobiographical record:
- 27-dimensional discrepancy feature

Record encoder:
- 27 -> 64 -> 64

Chronological autobiographical GRU:
- 64 -> hidden96

Output:
- predicted reliability A
- predicted reliability B

The learned output is a compact autobiographical self-model:

> "Given the history of what I expected myself to become versus what actually
> happened, how reliable is each future commitment for this continuing self?"

## New split

TRAIN subjects:
- 0..1023

DEV subjects:
- 3000..3255

TEST subjects:
- 6000..6255, sealed

v0.32.0 never touches TEST.

## DEV gates

Prediction:
- reliability MAE <= .10
- preferred commitment accuracy >= .95

Causal autobiographical controls:
- normal history forms the correct commitment >= .95
- donor history flips the later commitment >= .95

Erasure:
- empty autobiography must yield exactly [0.5, 0.5] reliability
- >= .95 of probes must be exact-neutral

Expected-versus-actual discrepancy ablation:
- preserve episode count/order and A/B labels
- replace every expected trajectory with the corresponding actual trajectory
- all discrepancy information becomes zero
- strong A/B preference must disappear >= .95

Persistence:
- autobiographical memory features are bit-exact after save/reload
- reliability output is bit-exact after reload
- later commitment preference is bit-exact after reload

## Why the discrepancy ablation matters

A history model might learn a spurious preference from:
- episode order;
- A/B frequency;
- fixed subject structure.

The discrepancy ablation preserves all of those.

It removes only:

`expected future - actual future`

If preference disappears, the mismatch between expected-self and actual-self is
causally load-bearing.

## Present-state control

The history model receives no present world and no present self.

Therefore in donor-history tests the present state is effectively held fixed by
construction.

Changing only autobiography must change later commitment formation.

This is stronger than merely showing that past outcomes correlate with choice.

## Meaning of PASS

A PASS supports this narrow functional claim:

The agent can store expected personal futures before acting, store the personal
future that actually occurred, accumulate those discrepancies autobiographically,
and use that history to form later commitments. Replacing, erasing, or removing
expected-versus-actual discrepancy from autobiography causally changes the later
commitment while present-state information is unavailable to the learner.

This is a computational form of autobiographical self-model adaptation.

It does not establish subjective experience, phenomenal consciousness,
sentience, qualia, emotion, personal rights, or moral status.

## After DEV PASS

Run one frozen TEST on subjects 6000..6255.

If TEST holds, the next milestone should stop using a fixed hidden reliability
profile and introduce developmental change:

- the self-model changes behavior based on autobiography;
- that changed behavior alters later autobiography;
- the resulting autobiographical evidence updates the self-model again.

That closes a longer loop:

self-expectation
-> choice
-> experience
-> autobiography
-> changed self-model
-> changed future choice
-> new experience

Such a loop would move from passive self-history toward a self-model that
participates in its own development.

## After DEV FAIL

Do not touch TEST.

Classify the failure as:
- expected trajectory quality;
- autobiographical record representation;
- chronological integration;
- reliability inference;
- donor-history causality;
- discrepancy ablation;
- persistence.

Repair only inside v0.32 DEV.
