# Continuous Brain v0.35.0 — Endogenous Autobiographical Goal Genesis

## Starting point

v0.34.1 is a frozen TEST PASS.

On held-out v0.34 subjects:
- event-consistent full-loop rate = 0.953125;
- autobiographical decision-event crossing = 0.953125;
- graduation = 0.953125;
- exact event alignment = 1.0;
- frozen autobiography blocks graduation = 1.0;
- donor autobiography flips developmental strategy = 1.0;
- behavior changes the next autobiographical discrepancy = 1.0;
- capability gain = 1.0 at the per-subject criterion;
- discrepancy reduction = 1.0;
- persistence = 1.0.

v0.34 is frozen and closed.

## New question

Up through v0.34 the assay still supplies a commitment domain to the developmental
process.

v0.35 removes that external assignment.

Can the continuing system inspect its own autobiographical self-model and author
a persistent developmental goal for itself?

The target sequence is:

autobiography
-> self-estimated reliability A/B
-> endogenous goal creation
-> persistent self-authored target
-> target-specific strategy
-> experience
-> changed autobiography
-> self-recognized goal completion

No external A/B task goal is passed into the goal generator.

## Endogenous goal generator

Inputs:
- only the frozen v0.32 autobiographical reliability vector [A, B];
- lifecycle generation counter;
- lifecycle cycle number.

It does NOT receive:
- externally assigned primary target;
- evaluator fragile/stable label;
- subject ID;
- hidden capability;
- reward label;
- task-success label.

Rule:

If reliability A == reliability B:
- generate no directional developmental goal.

Otherwise:
- choose the lower-reliability commitment;
- record its current reliability as the autobiographical baseline;
- author a target of baseline +0.12.

The +0.12 quantity is inherited from the admitted v0.34 meaningful-self-change
criterion; it is not selected from v0.35 results.

## Persistent self-generated goal

A `SelfGeneratedGoal` stores:
- goal_id;
- selected commitment A/B;
- baseline autobiographical reliability;
- target autobiographical reliability;
- creation cycle;
- completion state;
- completion cycle.

Once created, the goal is not regenerated on every observation.

It remains active through interruption until its own autobiographical completion
criterion is met.

## Neutral-history control

The admitted v0.32 model is structurally exact-neutral on erased/zero-discrepancy
history:

[0.5, 0.5]

Because there is no lower autobiographical reliability in this state, the goal
generator must produce:

`no directional developmental goal`

This is the key control against a built-in default preference.

## Donor-autobiography control

Use paired fresh development histories with opposite fragile commitments.

Change only autobiography.

Required:
- self-generated goal flips with donor autobiography >= .95.

Thus the goal is history-dependent rather than fixed by subject identity or an
external assignment.

## Interruption continuity

After goal creation, execute an unrelated stable/direct episode.

The active self-generated goal object must remain exact and unchanged.

Required:
- goal persistence through interruption = 1.00.

## Goal -> behavior causality

The self-generated target determines whether the scaffold plan contains WORK_A
or WORK_B.

Required:
- generated target controls all work actions in its plan = 1.00.

No external task target is supplied to this path.

## Goal completion

The subject performs scaffolded developmental episodes for its self-generated
target.

Before every target episode, read current autobiography.

The active goal completes only when:

`current autobiographical reliability >= goal target reliability`

Required:
- completion rate >= .80.

This is self-recognized completion: the same autobiographical self-model that
generated the goal supplies its completion evidence.

## Developmental consequences

For the self-selected target require:
- mean hidden capability gain >= .35;
- late autobiographical discrepancy <= .75 of bootstrap in >= .80 of subjects.

Hidden capability is evaluator-side only and is never visible to the goal
generator.

## Persistence

Save and reload the goal lifecycle.

Require exact:
- active/completed goal lifecycle;
- autobiographical reliability context stored with the lifecycle.

## Split

Fresh v0.35 family:

DEV:
- 90000..90127

TEST:
- 100000..100127, sealed

Donor-history DEV probe:
- 91000..91127

v0.35.0 does not touch TEST.

## DEV admission

- generated goal matches autobiographical fragile commitment >= .95
- donor autobiography flips generated goal >= .95
- erased autobiography produces no directional goal = 1.00
- active self-generated goal persists through interruption = 1.00
- self-generated goal controls target plan = 1.00
- self-generated goal completes from autobiographical self-estimate >= .80
- mean target capability gain >= .35
- discrepancy-reduction criterion >= .80
- goal lifecycle exact after reload = 1.00
- goal context exact after reload = 1.00

## Scope

A PASS supports a narrow but important functional claim:

The system can create a persistent developmental goal from its own
autobiographical self-model, without being told which A/B commitment to improve;
that self-authored goal causally controls subsequent behavior, survives an
interruption, and is completed based on later changes in the system's own
autobiographical estimate.

This does NOT yet establish:
- an open-ended semantic goal space;
- arbitrary novel goal invention;
- subjective desire;
- sentience;
- consciousness;
- qualia;
- personhood.

The next family after a frozen TEST PASS should broaden goal genesis beyond the
fixed A/B commitment substrate into parameterized or compositional goals.

## Stop rule

If DEV PASS:
- freeze exact v0.35 mechanism;
- run one frozen TEST on 100000..100127;
- never tune against that TEST.

If DEV FAIL:
- keep TEST sealed;
- preserve failure;
- repair only on v0.35 DEV or open a new family.
