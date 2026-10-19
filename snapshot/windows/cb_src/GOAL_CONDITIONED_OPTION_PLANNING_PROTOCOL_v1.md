# Continuous Brain v0.20.0 — Goal-Conditioned Autonomous Option Planning

## Starting point

v0.19 remains closed.

v0.19.2 established on CPU that the CURRENT canonical child reproduces the
historical P4b planner exactly:

- direct learned options: completion 0.61, mean collisions 18.42
- hierarchical planner: completion 0.38, mean collisions 21.72
- reactive: completion 0.00, mean collisions 112.73

All legacy `planning.*`, `options.*`, and `target_spatial.*` tensors are
bit-identical to the historical v0.10.3 planner.

The same smoke protocol fails on DirectML despite identical tensors, so CPU is
the qualified reference backend for this branch.

## What v0.20 changes

NO learned tensor is changed.

Primitive control still belongs to the trained temporal option policy.

Only the high-level option-macro ranking is given an explicit goal vector.

Historical ranking:

`discounted imagined reward + terminal option-policy Q`

Goal-conditioned ranking adds:

`+ success_weight * predicted success probability`
`- collision_weight * predicted collision probability`
`+ goal_value_weight * learned goal_value`

Two deliberately distinct profiles are evaluated:

### completion
- success weight 0.75
- collision weight 0.15
- learned goal-value weight 0.25

### safety
- success weight 0.25
- collision weight 1.25
- learned goal-value weight 0.10

The terminal option-policy Q weight remains 1.0 in both.

These values are experimental planner preferences, not claims about optimal
utility.

## Autonomous trajectory requirement

Unlike the v0.19 synthetic response assay, the planner's selected primitive
actions are sent to `DetourWorld.step()`.

Those actions determine all subsequent observations.

No teacher trajectory is used.

## Causal goal intervention

At every learned option boundary, the same exact planning latent is evaluated
under BOTH goal profiles before the environment advances.

The report records:
- the option selected under the active goal;
- the option selected under the counterfactual goal;
- whether the goal swap changes the option;
- predicted reward/success/collision/goal-value components for both plans.

Therefore any option difference at that point is caused by changing only the
explicit goal weights.

## Evaluation seeds

Seeds 13001-13020 are not in the historical P4b report seed sets.

This does NOT establish that those seeds were absent from every historical
training dataset, so the package does not claim a fresh generalization result.

## No scientific threshold yet

This first run is a causal capability diagnostic.

Software PASS means:
- evaluation completed;
- actions were autonomous;
- no protected artifact changed;
- no training occurred;
- v0.19 remained closed.

After seeing the observed goal-swap rate and trajectory consequences, freeze an
admission protocol before further development.

## Next step

If goal swaps produce meaningful high-level policy changes, connect the admitted
canonical memory system to an explicit goal register so remembered information
can select or parameterize the planning goal.

If changing the explicit goal rarely changes option choice, improve the
high-level goal-conditioned planner before involving memory.
