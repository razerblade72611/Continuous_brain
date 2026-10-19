# Continuous Brain v0.30.0 — Joint Self / World Prediction

## Why v0.30 is a new family

v0.29.3 is a frozen TEST FAIL and remains permanently recorded as such.

Its central categorical/causal results generalized:
- policy: PASS
- retrieval: PASS
- goal: PASS
- planning mode: PASS
- identity: PASS
- donor-self intervention: PASS
- autobiographical donor/erase intervention: PASS
- existing episodic-reader routing: PASS
- broadcast-off exact historical preservation: PASS
- persistent reload: PASS

The only failed TEST gate was:

`history_repaired_priority_mae = 0.05986 > 0.05`

Because the broadcast TEST is spent, v0.29 is closed.  Nothing in v0.30 is
trained against that TEST.

The architectural lesson is that a compact shared broadcast can preserve
categorical self structure while still losing some fine-grained continuous
self-state information under autobiographical state transitions.

v0.30 therefore does not add another priority readout.

It creates a new development family in which persistent self variables are
treated as first-class dynamical state that must be predicted through time.

## Central question

For a candidate action, can the model predict both:

1. what happens to the external world; and
2. what happens to the continuing internal self?

The predicted self includes:
- unresolved commitment A;
- unresolved commitment B;
- confidence;
- uncertainty;
- world-attributed prediction error;
- agency estimate;
- next active goal.

This supports a stronger counterfactual:

> "If I choose this action, what will happen in the world, and what will happen
> to my continuing internal state?"

## Joint model

Input:
- current world state: 4 values
- current persistent-self vector: 10 values
- candidate action: 4-way one-hot

Shared latent:
- 96 dimensions

Outputs:
- next world4
- next continuous self6
- next goal logits4
- agency probability
- auxiliary reconstruction of current self10

Only this new model is trained.

Frozen:
- historical InfantBrain
- admitted working memory
- v0.23 episodic reader
- v0.28 persistent-self mechanism
- all historical planning/consequence models
- the closed v0.29 results

## World variables

The new cognitive development world is intentionally abstract rather than a
robotics task.

World state:
- progress A
- progress B
- information quality
- volatility

Actions:
- OBSERVE
- WORK_A
- WORK_B
- WAIT

The purpose is to isolate continuing-self dynamics, not motor skill.

## Self/world decoupling

Some TRAIN states have saturated external progress.

In those states:
- WORK_A and WORK_B can have identical visible world progress outcomes;
- the actions can still have different consequences for unresolved internal
  commitments.

This is deliberate.

It creates cases in which action choice cannot be justified only by external
world progress.  The system must evaluate future self-state.

## New split

This is a new family unrelated to the spent v0.29 TEST IDs.

TRAIN:
- 0..19999

DEV:
- 40000..44999

TEST:
- 60000..64999, sealed

v0.30.0 does not touch TEST.

## Prediction gates

DEV requirements:
- next-world MAE <= .025
- next continuous-self MAE <= .030
- next-goal accuracy >= .98
- agency accuracy >= .98
- current-self reconstruction MAE <= .035

The reconstruction objective ensures current self information remains recoverable
inside the joint latent rather than being discarded as an irrelevant input.

## Counterfactual self-consequence assay

External world is set to a saturated state:
- progress A = 1
- progress B = 1

Thus visible progress cannot distinguish WORK_A from WORK_B.

Two persistent selves are supplied with the exact same world:
- self A has a strong unresolved A commitment;
- self B has a strong unresolved B commitment.

Full planner:
- evaluates predicted world consequence + predicted self consequence.

Required:
- self A chooses WORK_A;
- self B chooses WORK_B;
- accuracy >= .95.

Self-consequence ablation:
- identical model predictions;
- remove only the future-self term from utility.

Required:
- correct action accuracy <= .35.

Donor-self intervention:
- external world fixed;
- replace only current self A with self B.

Required:
- chosen action follows donor self >= .95.

This is a direct test that predicted future self-state, not only external world
state, is causally load-bearing in counterfactual choice.

## Predicted-self error assay

The model predicts the future self under WORK_A.

Normal reality:
- work succeeds;
- future self remains commitment-directed.

Surprise reality:
- the external intervention causes the work to fail;
- confidence falls;
- uncertainty rises;
- prediction-error state rises;
- agency attribution shifts away from self.

Compare:
- predicted future self
- actual future self

That difference is the metacognitive self-prediction error.

Required:
- mean surprise self-prediction error >= .10;
- surprise recruits INVESTIGATE >= .95;
- normal transition recruits INVESTIGATE <= .10.

Causal control:
- keep the surprise history/commitment but replace only uncertainty and
  prediction-error fields with the model-predicted self values.

Required:
- INVESTIGATE rate returns <= .10.

This tests whether deviation from the predicted self-state can recruit an
information-seeking mode.

## Persistence

A continuing self is saved and reloaded between candidate-action evaluations.

Required:
- action choice before and after reload is identical = 1.00.

Thus the counterfactual self-model is conditioned on the same persistent state
that survived the v0.28 continuity tests.

## Meaning of PASS

PASS supports this narrow functional claim:

The system can predict external and internal consequences of candidate actions
jointly.  Predicted future self-state causally changes counterfactual action
selection even when visible world progress is held fixed.  Unexpected divergence
between predicted self and actual self recruits information-seeking, and the
choice process survives persistent save/reload.

This is a computational self-model result.

It does not establish subjective experience, phenomenal consciousness,
sentience, qualia, emotion, desire, or moral status.

## After DEV PASS

Run one frozen TEST on 60000..64999.

If TEST holds, the next milestone should extend from one-step self/world futures
to recurrent multi-step self trajectories:

`current self -> imagined future self1 -> future self2 -> ...`

Planning would then compare whole possible personal trajectories, not merely the
next internal state.

Autobiographical memory should record which imagined self-trajectory was chosen
and later compare it with the self that actually emerged.

## After DEV FAIL

Do not touch TEST.

Classify the failure as:
- world dynamics;
- self dynamics;
- goal transition;
- agency prediction;
- latent self preservation;
- counterfactual utility coupling;
- self-prediction-error recruitment.

Repair only within the new v0.30 DEV family.
