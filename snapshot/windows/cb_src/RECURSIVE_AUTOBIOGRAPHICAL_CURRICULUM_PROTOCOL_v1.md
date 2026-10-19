# Continuous Brain v0.36.0 — Recursive Autobiographical Curriculum

## Starting point

v0.35.3 is a frozen TEST PASS.

On held-out subjects, v0.35 established that autobiography alone can author one
bounded developmental target:
- target selection = 1.00;
- donor autobiography flips target = 1.00;
- erased autobiography produces no directional target = 1.00;
- active goal survives interruption = 1.00;
- target controls behavior = 1.00;
- self-recognized completion = 0.9140625;
- capability and discrepancy gates pass;
- lifecycle persistence = 1.00.

v0.35 is closed.

## New question

Can the system's changed autobiography decide what it should develop next?

The sequence is:

bootstrap autobiography
-> self-authored goal 1
-> persistent goal-directed development
-> self-recognized completion
-> changed autobiography
-> self-authored goal 2
-> new goal-directed behavior

No external second target is supplied.

## Why this matters

v0.35 demonstrated endogenous goal genesis once.

A continuing self-directed learner must be able to form a developmental
curriculum over time rather than requiring an external experimenter to assign a
new goal after every completion.

v0.36 tests recursive goal authorship.

## First goal

Reuse the exact admitted v0.35 mechanism:

- goal source: frozen autobiographical reliability;
- lower-reliability commitment is selected;
- exactly neutral history generates no directional goal;
- target = current autobiographical reliability +0.12;
- active goal survives unrelated interruption;
- maximum first-goal horizon = 16 episodes;
- completion uses the same frozen autobiographical self-estimate.

## Second goal

After goal 1 completes:

1. read the updated autobiography;
2. run the exact same endogenous goal generator;
3. create goal 2 without any external A/B assignment.

A second goal counts as **curriculum advancement** when:

- its primary commitment differs from goal 1; OR
- it keeps the same primary but its target reliability is at least 0.05 higher
  than goal 1's target.

The 0.05 advancement requirement is fixed before v0.36 DEV.

## Frozen-autobiography counterfactual

For the same subject, after goal 1 has completed:

- keep actual capability and real developmental history untouched;
- but generate the counterfactual goal 2 from the ORIGINAL bootstrap
  autobiography instead of the updated autobiography.

Required result:

- frozen bootstrap autobiography reproduces the semantic content of goal 1
  (same primary, same baseline, same target) >= .95;
- updated autobiography produces a different/advanced goal >= .95.

This is the central causal assay.

It tests whether the curriculum advances because autobiography changed, rather
than merely because a second goal-generation call occurred.

## Second-goal behavior

The updated second goal must determine whether the developmental plan targets A
or B.

Then execute four scaffolded probe episodes for goal 2.

Require:
- second-goal plan causality = 1.00;
- target capability changes in >= .80 of first-goal completers.

The four-episode probe is not a second-goal completion test. Its purpose is only
to establish that the recursively authored goal is behaviorally operative.

## Donor and erased controls

Fresh donor histories:
- changing autobiography must flip first-goal identity >= .95.

Erased history:
- no directional developmental goal = 1.00.

## Persistence

Persist:
- first goal;
- updated second goal;
- frozen-history counterfactual second goal.

Require exact save/reload equality = 1.00.

## Fresh split

DEV:
- 110000..110127

TEST:
- 120000..120127, sealed

Donor probe:
- 111000..111127

v0.36.0 does not touch TEST.

## DEV gates

- first endogenous goal matches autobiographical fragility >= .95;
- first goal completes >= .80;
- updated autobiography authors a second goal >= .95 of first-goal completers;
- updated autobiography advances the curriculum >= .95;
- frozen bootstrap autobiography repeats old goal semantics >= .95;
- updated vs frozen second-goal semantics diverge >= .95;
- donor history changes first goal >= .95;
- erased history produces no directional goal = 1.00;
- second goal controls behavior = 1.00;
- second-goal probe changes target capability >= .80;
- curriculum snapshot persistence = 1.00.

## Scope

A PASS would support a bounded functional claim:

The system can use the consequences of one self-authored developmental episode
to update its autobiography, and that changed autobiography can author a
different or more demanding next developmental goal. Holding autobiography
frozen at its earlier state prevents curriculum advancement.

This is recursive autobiographical curriculum formation.

It does NOT establish:
- open-ended semantic goal invention;
- arbitrary symbolic goal composition;
- subjective desire;
- sentience;
- consciousness;
- qualia;
- personhood.

A later family should expand the goal representation beyond the current A/B
commitment substrate.

## Stop rule

If DEV PASS:
- freeze exact mechanism and thresholds;
- run one frozen TEST on 120000..120127.

If DEV FAIL:
- preserve failure;
- keep TEST sealed;
- diagnose on DEV only.
