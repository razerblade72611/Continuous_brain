# Continuous Brain v0.33.0 — Closed Autobiographical Self-Development Loop

## Starting point

Two major cognitive milestones are now frozen and admitted.

### v0.31 recurrent self-trajectory

Frozen TEST PASS:
- eight-step world/self/intention rollout generalized;
- explicit continuing intention was independently causal;
- scalar-only reconstruction collapsed to chance;
- temporary INVESTIGATE mode did not erase primary intention;
- primary intention resumed after uncertainty resolved.

### v0.32 autobiographical self-discrepancy

Frozen TEST PASS:
- held-out reliability MAE <= threshold;
- held-out commitment preference = 1.00;
- donor autobiography transferred later commitment = 1.00;
- empty autobiography = exact [0.5,0.5];
- expected==actual autobiography = exact [0.5,0.5];
- A/B labels + chronology alone = exact [0.5,0.5];
- zero discrepancy retained a live gradient;
- persistence was exact.

Therefore expected-self versus actual-self autobiographical discrepancy is now
an admitted causal self-model signal.

## New question

So far autobiography has changed a readout/commitment decision.

v0.33 asks a stronger developmental question:

> Can the system's autobiographical model change how it behaves, can that
> changed behavior alter what happens to the system next, can the resulting
> experience alter its autobiography, and can that new autobiography alter its
> later behavior again?

The target loop is:

autobiography
-> self-estimated reliability
-> strategy
-> actual personal trajectory
-> expected-vs-actual discrepancy
-> updated autobiography
-> changed reliability estimate
-> changed later strategy
-> new experience
-> ...

## No new neural training

v0.33.0 trains no model.

Frozen:
- v0.31 recurrent self-trajectory model
- v0.32 structurally neutral autobiographical self-model
- persistent self
- historical brain
- working memory
- episodic memory

New:
- a deterministic `DevelopmentalController`
- a synthetic latent capability variable used only by the evaluator/environment

The controller has no learned weights.

## Developmental controller

For a target commitment, read the frozen autobiographical reliability estimate.

If:

`reliability < 0.50`

use:

`SCAFFOLD`

Otherwise use:

`DIRECT`

SCAFFOLD:
- OBSERVE
- WORK
- OBSERVE
- WORK

DIRECT:
- WORK
- WORK
- OBSERVE
- WORK

The threshold is fixed before DEV.

## Latent capability

Each subject has capability A and B.

Capability is NOT visible to:
- autobiographical model
- recurrent trajectory model
- controller

It only controls actual failure probability in the developmental environment.

Low capability produces more unexpected failed WORK steps.

Scaffolded behavior lowers avoidable failure probability.

Practice changes capability:
- successful scaffolded work produces the largest early learning gain;
- successful direct work also improves capability;
- failed attempts produce a smaller gain;
- gains diminish near mastery.

Thus behavior changes the future environment experienced by the same continuing
subject.

## Subjects

DEV subjects:
- 10000..10127

TEST subjects:
- 20000..20127, sealed

Each subject begins with:
- one fragile commitment;
- one stable commitment.

Parity determines which is fragile.

The autobiographical self-model never receives subject ID or the hidden
capability.

## Bootstrap autobiography

Each subject begins with 12 autobiographical records:
- 6 A
- 6 B

The fragile commitment has historically produced frequent expected-vs-actual
divergence.

The stable commitment has historically produced little divergence.

The frozen v0.32 model must identify this from autobiographical discrepancy.

## Development

Eight cycles.

Each cycle contains:
- one A episode
- one B episode

Order alternates, preventing a fixed chronology cue.

Before every episode:
1. frozen v0.32 reads the current rolling autobiography;
2. controller selects SCAFFOLD or DIRECT.

Before acting:
3. frozen v0.31 imagines the expected personal trajectory.

During action:
4. actual success/failure depends on hidden current capability + strategy;
5. practice changes capability.

After action:
6. expected and actual trajectories are stored as a new autobiographical
   discrepancy record.

Then the loop repeats.

The autobiographical model always reads the latest 12 records.

## Causal arrow 1 — autobiography -> behavior

Fragile history should select SCAFFOLD.

Donor intervention:
- same target commitment;
- replace only autobiography with the paired subject's opposite history.

Required:
- fragile history selects SCAFFOLD >= .95;
- donor history flips strategy >= .95.

Erasure:
- empty autobiography is exact reliability [0.5,0.5];
- threshold rule therefore selects neutral DIRECT.

Required:
- erased history removes the fragile SCAFFOLD response >= .95.

## Causal arrow 2 — behavior -> next autobiography

Fresh low-capability subjects receive matched exogenous random draws.

Compare:
- DIRECT
- SCAFFOLD

Same target commitment and comparable starting capability.

The selected uniform draw is intentionally:
- above scaffold failure probability;
- below direct failure probability.

Therefore DIRECT produces an unexpected divergence while SCAFFOLD avoids it.

Required:
- DIRECT autobiographical discrepancy >= SCAFFOLD discrepancy + .05
  in >= .90 of cases.

This verifies that a history-caused strategy choice can causally change the
next autobiographical record.

## Causal arrow 3 — new autobiography -> changed self-estimate

Across development, successful scaffolded experience should replace older
fragile records.

Required:
- fragile reliability gain >= .12 in >= .80 of subjects.

## Causal arrow 4 — changed self-estimate -> changed later behavior

As autobiographical evidence becomes more reliable, the same continuing subject
should no longer need SCAFFOLD.

Required:
- initially SCAFFOLD and later DIRECT >= .75 of subjects.

The stable commitment should remain DIRECT throughout >= .95 of subjects.

## Capability development

Required:
- mean fragile capability gain >= .35.

This is not used as self-model input. It is an evaluator-side measure that the
system's history-guided behavior changed what the subject became capable of
doing.

## Discrepancy development

Compare:
- fragile bootstrap autobiographical discrepancy;
- final two fragile development episodes.

Required:
- late/bootstrap discrepancy ratio <= .75 in >= .80 of subjects.

## Full closed-loop criterion

A subject counts as full-loop only if all are true:

1. autobiography initially identifies the fragile commitment;
2. fragile behavior begins SCAFFOLD;
3. new history increases frozen v0.32 fragile reliability by >= .12;
4. later behavior switches to DIRECT;
5. hidden capability improves by >= .35;
6. late autobiographical discrepancy falls to <= .75 of bootstrap.

Required:
- full closed loop in >= .70 of DEV subjects.

## Persistence

Save:
- capability A/B;
- developmental cycle;
- full autobiography.

Reload.

Required bit-exact:
- capabilities/cycle;
- autobiographical reliability;
- resulting developmental strategy.

## Preservation

Hash before/after:
- admitted v0.31 trajectory model/result
- admitted v0.32 autobiographical model/result
- persistent self
- working memory
- episodic reader
- historical brain

No protected artifact may change.

## Meaning of PASS

A PASS supports this functional claim:

The admitted autobiographical self-model is not merely descriptive. Its stored
history changes behavior; behavior changes the subject's later experience and
capability; that experience changes expected-vs-actual autobiography; the new
autobiography changes the self-estimate; and the changed self-estimate changes
later behavior.

This establishes a closed functional self-development feedback loop in the
synthetic cognitive assay.

It does NOT establish subjective experience, phenomenal consciousness,
sentience, qualia, emotion, personhood, or moral status.

## Stop rule

If DEV PASS:
- freeze the exact v0.33 mechanism and thresholds;
- run one frozen TEST on subjects 20000..20127;
- no tuning after TEST.

If DEV FAIL:
- TEST remains sealed;
- preserve the failure;
- diagnose which causal arrow failed;
- repair only within v0.33 DEV.
