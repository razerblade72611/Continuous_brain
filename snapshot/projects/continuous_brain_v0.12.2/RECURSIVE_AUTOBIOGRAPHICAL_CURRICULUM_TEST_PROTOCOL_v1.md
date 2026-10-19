# Continuous Brain v0.36.1 — Frozen Recursive Autobiographical Curriculum TEST

## Authorization

v0.36.0 DEV PASS_CONFIRMED on fresh subjects 110000..110127.

Observed DEV:
- first endogenous goal genesis = 1.00
- first-goal completion = 0.859375
- second-goal generation given first completion = 1.00
- curriculum advancement given first completion = 1.00
- frozen bootstrap autobiography repeats first-goal semantics = 1.00
- updated vs frozen second-goal divergence = 1.00
- second-goal plan causality = 1.00
- second-goal target-capability change = 1.00
- donor-history first-goal flip = 1.00
- erased-history neutral/no-goal = 1.00
- curriculum persistence = 1.00

v0.35.3 remains a frozen TEST PASS and its TEST is closed.

## Frozen mechanism

No training.

No changes from v0.36.0:

- frozen v0.31 recurrent self-trajectory model;
- frozen v0.32 autobiographical self-model;
- admitted v0.35 endogenous goal generator;
- first developmental goal comes only from bootstrap autobiography;
- first-goal target = autobiographical baseline +0.12;
- first-goal maximum horizon = 16;
- after first-goal completion, goal 2 is generated from UPDATED autobiography;
- no external second goal is supplied;
- curriculum advancement means:
  - different primary commitment, OR
  - same primary with target >= previous target +0.05;
- frozen-bootstrap counterfactual regenerates goal 2 from ORIGINAL autobiography;
- second-goal behavioral probe = 4 scaffolded episodes;
- all admission thresholds remain unchanged.

## Frozen TEST population

- 120000..120127

This population was sealed during v0.36 DEV.

It is opened exactly once by this package.

After this run it is permanently spent regardless of PASS or FAIL.

## Central causal test

For each held-out first-goal completer:

### Updated-history path
Use the actual autobiography after self-authored goal 1 has been pursued and
completed.

Generate goal 2 from that updated autobiography.

### Frozen-history path
Keep the actual continuing subject and developmental state, but feed the goal
generator the ORIGINAL bootstrap autobiography.

Generate the counterfactual goal 2.

Required:

- frozen bootstrap history reproduces semantic content of goal 1 >= .95;
- updated autobiography authors goal 2 >= .95;
- updated goal 2 advances the curriculum >= .95;
- updated and frozen goal-2 semantics diverge >= .95.

This is the decisive test that curriculum advancement depends on changed
autobiography rather than merely on calling the goal generator a second time.

## Held-out donor/neutral controls

The TEST itself supplies the donor histories.

Neighboring even/odd held-out histories have opposite autobiographical
fragility.

Changing only autobiographical history must flip first-goal identity >= .95.

Erased/empty autobiography must produce no directional goal = 1.00.

No DEV donor IDs are reused.

## Behavioral consequence

The recursively authored second goal must determine whether the developmental
plan targets A or B.

Then run the frozen four-episode scaffolded second-goal probe.

Required:

- second-goal plan causality = 1.00;
- target capability changes in >= .80 of first-goal completers.

This is not a second-goal completion test.

## Persistence

Persist:
- first self-authored goal;
- updated-history second goal;
- frozen-bootstrap counterfactual second goal.

Require exact snapshot equality after reload = 1.00.

## Admission thresholds

Frozen from v0.36.0:

- first endogenous goal matches autobiographical fragility >= .95;
- first goal completes >= .80;
- updated autobiography authors second goal >= .95 conditional on first completion;
- updated autobiography advances curriculum >= .95;
- frozen bootstrap autobiography repeats old goal semantics >= .95;
- updated vs frozen second-goal semantics diverge >= .95;
- donor autobiography changes first goal >= .95;
- erased autobiography has no directional goal = 1.00;
- second goal controls behavior = 1.00;
- second-goal probe changes target capability >= .80;
- curriculum snapshot persistence = 1.00.

## Meaning of PASS

A frozen TEST PASS supports the following bounded functional result on unseen
subjects:

A self-authored developmental goal changes the continuing system's experience
and autobiography. After that goal is recognized as complete, the updated
autobiography can recursively author a different or more demanding next goal
without an external second assignment. Holding autobiographical input frozen at
the earlier bootstrap state prevents that curriculum advancement and reproduces
the earlier goal instead.

That is functional recursive autobiographical curriculum formation.

It does not establish:
- open-ended semantic goal invention;
- arbitrary symbolic goal composition;
- subjective desire;
- sentience;
- phenomenal consciousness;
- qualia;
- personhood.

## Stop rule

After this TEST:
- 120000..120127 are permanently spent;
- never tune v0.36 against them.

If PASS:
- close/freeze v0.36;
- the next family should expand the goal representation beyond the fixed A/B
  substrate into a parameterized or compositional self-authored goal space.

If FAIL:
- preserve the frozen TEST FAIL;
- close v0.36 TEST;
- open a new development family instead of tuning against TEST.
