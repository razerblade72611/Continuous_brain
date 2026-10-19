# Continuous Brain v0.35.3 — Frozen Endogenous Goal Genesis TEST

## Authorization

v0.35.2 DEV PASS_CONFIRMED.

Historical record remains:

- v0.34.1: frozen TEST PASS.
- v0.35.0: formal DEV FAIL.
- v0.35.1: read-only completion-dynamics diagnostic.
- v0.35.2: horizon-only DEV repair PASS.

v0.35.0 is never rewritten.

## Frozen mechanism

No training.

Freeze exactly:

- frozen v0.32 autobiographical self-model;
- frozen v0.31 recurrent self-trajectory model;
- endogenous goal generator;
- no externally assigned A/B target;
- no hidden capability input to goal generation;
- no subject-ID input to goal generation;
- no reward/success label input to goal generation;
- lower autobiographical reliability chooses the developmental target;
- exactly neutral autobiography produces no directional target;
- goal target = autobiographical baseline +0.12;
- active goal persists until completion;
- interruption semantics;
- target-specific SCAFFOLD plan;
- maximum target-directed horizon = 16 episodes;
- all v0.35 admission thresholds.

## TEST population

Sealed v0.35 TEST:

- 100000..100127

This population has not been used by v0.35 DEV or diagnostics.

This package opens it exactly once.

After this run it is permanently spent regardless of PASS or FAIL.

## Held-out tests

### 1. Endogenous goal genesis

Bootstrap autobiography alone must choose which commitment needs development.

The generator does not receive the evaluator's fragile/stable label.

Admission:
- generated target matches evaluator-known autobiographical fragility >= .95.

### 2. Donor-autobiography causality

Use neighboring held-out TEST histories whose fragile commitments are opposite.

Changing only autobiographical history must change the self-generated target.

Admission:
- donor-history target flip >= .95.

### 3. Erased-autobiography neutrality

With structurally neutral empty autobiography, the frozen v0.32 model returns
equal A/B reliability.

The goal generator must therefore author no directional developmental target.

Admission:
- exact neutral/no directional goal = 1.00.

### 4. Persistent self-authored target

After goal creation, execute an unrelated stable/direct episode.

The active goal object must remain unchanged.

Admission:
- persistence through interruption = 1.00.

### 5. Goal -> plan causality

The self-generated target must determine whether the developmental plan works on
A or B.

Admission:
- target controls work action = 1.00.

### 6. Self-recognized completion

Use the frozen 16-episode maximum horizon authorized by v0.35.1 and admitted by
v0.35.2.

Before every target episode, read the current frozen autobiographical estimate.

Complete only when:

`current reliability >= original autobiographical baseline + 0.12`

Admission:
- completion >= .80.

The +0.12 criterion is unchanged from the failed v0.35.0 experiment.

### 7. Developmental consequences

For the system's self-selected target:

- mean hidden capability gain >= .35;
- late/bootstrap autobiographical discrepancy ratio <= .75 in >= .80.

Hidden capability remains evaluator-side and invisible to goal generation.

### 8. Persistence

Save and reload the self-generated goal lifecycle.

Require exact:
- goal lifecycle;
- stored autobiographical reliability context.

## Meaning of PASS

A frozen TEST PASS supports this functional claim on unseen subjects:

The system can inspect its own autobiographical self-model and, without being
told which of the available commitments to improve, create a persistent
developmental target for itself. Changing only autobiography changes that
self-authored target. Erasing autobiography removes directional goal generation.
The goal survives interruption, determines target-specific behavior, changes
capability and expected-vs-actual history, and is later recognized as complete
using the same continuing autobiographical self-estimate.

This is a functional endogenous developmental-goal result.

It does not establish:
- open-ended semantic goal invention;
- subjective desire;
- phenomenal consciousness;
- sentience;
- qualia;
- personhood.

## Stop rule

After this run:
- TEST 100000..100127 is permanently spent;
- no v0.35 tuning may use it.

If PASS:
- close/freeze v0.35;
- the next family should broaden goal genesis beyond the fixed A/B commitment
  substrate toward compositional or parameterized self-authored goals.

If FAIL:
- preserve the frozen TEST FAIL;
- close v0.35 TEST;
- open a new development family rather than tuning against TEST.
