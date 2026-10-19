# Continuous Brain v0.33.1 — History-Relative Developmental Hysteresis Repair

## Source status

v0.33.0 remains a formal DEV FAIL.

It successfully demonstrated:
- autobiography initially identified the fragile commitment: 1.00
- fragile autobiography selected SCAFFOLD: 1.00
- donor autobiography flipped strategy: 1.00
- erased autobiography removed the fragile response: 1.00
- strategy changed the next autobiographical discrepancy: 1.00
- fragile autobiographical reliability increased by >= .12 in 0.9375
- mean fragile capability gain: +0.679
- autobiographical discrepancy fell to ~0.101 of bootstrap
- persistence: 1.00

But:
- stable commitment stayed DIRECT in only 0.625
- fragile commitment switched to DIRECT in only 0.28125
- full loop rate therefore remained 0.28125

## Diagnosis

The failing controller used:

`DIRECT iff autobiographical reliability >= 0.50`

That conflated two concepts:

1. v0.32's **historical reliability calibration**
2. v0.33's **developmental readiness**

The v0.32 model was trained to distinguish histories resembling roughly:
- unreliable: ~0.05–0.15
- reliable: ~0.90–0.95

It was not trained to make `0.50` a universal developmental readiness boundary.

v0.33.0 proved the problem empirically:

A fragile commitment could:
- increase autobiographical reliability by +0.323 on average;
- increase capability by +0.679;
- reduce discrepancy by ~90%;

yet still fail to cross an absolute 0.50 reliability threshold.

The loop was changing the subject, but the controller was using the wrong
coordinate system to recognize that change.

## Repair principle

Use the commitment's own autobiographical baseline.

At bootstrap:

- reliability >= .50 -> DIRECT
- reliability < .50 -> SCAFFOLD

A scaffolded commitment graduates to DIRECT when:

`current reliability >= baseline reliability + 0.12`

The `0.12` value is not selected from the v0.33.0 outcome.

It is exactly the already-preregistered v0.33.0:

`MIN_RELIABILITY_GAIN = 0.12`

which defined a meaningful autobiographical self-estimate update before the
v0.33.0 DEV run.

This makes the repair history-relative rather than threshold-fitted.

## One-way developmental hysteresis

Once a commitment graduates, it remains DIRECT for this developmental assay.

That prevents a few later noisy discrepancy records from making a developed
commitment repeatedly oscillate between developmental phases.

This is intentionally narrower than a future lifelong adaptive controller,
which should later gain principled regression/recovery dynamics.

## Strong new causal control

The repair adds a frozen-autobiography counterfactual.

Run the same developmental environment:
- same starting capabilities;
- same practice;
- same actual experiences;
- capability may improve;

but make every strategy decision using only the original bootstrap
autobiography.

Thus the self-estimate available to the controller never changes.

Required:

`frozen autobiography blocks graduation >= .95`

This distinguishes:

`practice/cycle count -> behavior change`

from:

`new autobiography -> changed self-estimate -> behavior change`

If capability improves but behavior does not graduate while decision-history is
frozen, then the behavioral phase transition genuinely requires autobiographical
updating.

## Preserved causal arrows

v0.33.1 retains all original controls:

1. autobiography -> initial behavior
2. donor autobiography -> behavior flip
3. erased autobiography -> neutral DIRECT
4. behavior -> next autobiographical discrepancy
5. new autobiography -> changed frozen v0.32 self-estimate
6. changed self-estimate -> later strategy
7. behavior -> capability development
8. later development -> reduced autobiographical discrepancy
9. persistence across save/reload

## Data use

No neural training.

Reuse:
- DEV subjects 10000..10127

TEST remains sealed:
- 20000..20127

No new development window is consumed.

## Admission thresholds

Retain v0.33.0 thresholds:
- initial fragility identification >= .95
- initial fragile SCAFFOLD >= .95
- stable commitment always DIRECT >= .95
- donor-history behavior flip >= .95
- erased-history neutral behavior >= .95
- behavior changes next autobiography >= .90
- reliability gain >= .12 in >= .80
- fragile behavior switches to DIRECT >= .75
- mean capability gain >= .35
- discrepancy reduction criterion in >= .80
- full closed loop >= .70
- persistence = 1.00

New:
- frozen autobiography blocks graduation >= .95

## Meaning of PASS

A PASS supports the stronger functional statement:

The developmental transition is not caused merely by elapsed cycles or hidden
capability improvement. The subject must accumulate new autobiographical
evidence, the frozen v0.32 self-model must interpret that history as a meaningful
improvement relative to its own prior self-estimate, and that change must cause
a later behavioral phase transition.

This closes:

autobiography
-> self-estimate
-> scaffolded behavior
-> changed experience/capability
-> changed autobiography
-> improved self-estimate
-> direct behavior

while a frozen-autobiography counterfactual blocks the final transition.

This remains a computational developmental mechanism. It does not establish
subjective experience, consciousness, sentience, qualia, personhood, or moral
status.

## Stop rule

If DEV PASS:
- freeze v0.33.1;
- run one frozen TEST on subjects 20000..20127;
- never tune against that TEST.

If DEV FAIL:
- keep TEST sealed;
- preserve both v0.33.0 and v0.33.1 outcomes;
- diagnose the remaining causal arrow without changing admitted v0.31/v0.32.
