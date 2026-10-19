# Proposed next experiment: does existing imagination improve option choice?

Status: a protocol proposal, not a completed experiment or new controller.

## Question

At the same saved decision state, can the existing learned planner choose an
option that improves eventual outcome over the direct selector? Separate two
questions: whether a beneficial alternative exists, and whether predictions
identify it before observing the outcome.

## Fixed scope

- Reuse the same 80 recorded boundaries on already-spent seeds 14001–14015.
- Preserve the current checkpoint, direct selector, option action policies,
  termination policy, and all protected artifacts.
- No training, goal-weight tuning, memory connection, fresh confirmation seeds,
  or revival of v0.19 during this diagnosis.
- Hash the actual planner implementation used for the failed comparison and
  record the checkpoint it loads. Capture its unmodified scores and choices.
- Reject missing rows, incomplete branches, mismatched hashes, and stale
  reports. Mark incomplete computations as errors, not capability failures or
  successes. Keep raw per-step records and a unique run identifier.

## One intervention followed by a common controller

For every boundary, reconstruct its exact environment and controller state.
For each of the six possible initial options:

1. Execute that option using the existing action policy and actual execution
   contract: minimum commitment and learned termination.
2. After that option terminates, use the same frozen direct-option controller
   for all remaining decisions, responding to each branch's own observations.
3. Stop at completion or the original absolute episode deadline of 120 steps.
   Do not grant every branch another 120 steps from the boundary.

If an active option never terminates, it keeps control until the episode ends,
as in the current execution contract. The v0.20.3 12-step observation cap must
not silently become a new controller timeout.

All branches share the same continuation *policy*, not an open-loop replay of
the original episode's future actions. Distinct actions lead to distinct
observations and require closed-loop responses. This controls the rest of the
decision process while allowing different option durations to incur their real
cost within one common episode budget.

If a future controller is intended to execute only a fixed option prefix and
then replan, test that execution contract as a separately named arm. Do not
change it inside this comparison or mix its results with learned termination.

## Record predictions before executing any branch

Capture every score the existing planner actually uses, its selected option,
imagined actions, predicted rewards/collisions/success events, termination
predictions, rollout depth, and terminal value contribution. Do not let actual
counterfactual outcomes enter these predictions.

Check what each prediction head was trained to mean. A success logit trained
against a transition's completion event must not be interpreted as a calibrated
probability of eventual episode success. Likewise, distinguish a discounted
return estimate from an unscaled goal score. Compare corresponding targets,
horizons, and discounts.

For diagnosis, compare:

- One-step predictions conditioned on actual observations and executed actions.
- Multi-step predictions rolled forward from the planner's own predicted
  states, to expose additional rollout drift and action divergence.
- Final candidate ranking against full-remainder realized outcomes, to measure
  whether prediction errors actually change the chosen option for the worse.

This identifies whether the problem lies in immediate predictions, accumulated
rollout error, terminal scoring, termination prediction, or their combination.
These remain competing explanations until measured.

## Comparators and primary outcomes

Report completion first, then collisions, then steps to successful completion.
Keep shaped reward as a separate diagnostic. Avoid hiding these outcomes in a
new weighted sum.

Compare the full remainder after the first choice made by:

1. The frozen direct selector.
2. The existing learned planner, using only information available before acting.
3. An oracle selecting the best of the top two realized initial interventions.
4. An oracle selecting the best of all six realized initial interventions.

The oracle comparators use future information only for diagnosis. Each changes
one initial option and then uses the common direct continuation; they are not
deployable autonomous policies or repeated oracle replanners.

Report paired gains/losses at every boundary, then aggregate with episode/seed
clustering. A simple boundary average overweights episodes that generate more
boundaries. Include per-seed summaries and paired uncertainty; do not treat the
480 alternatives as independent trials.

## Decisions after this one experiment

| Evidence | Next action |
| --- | --- |
| Neither oracle improves useful outcomes over direct control on these states | Broader search is not justified by this sample; investigate skill/termination failures and missing informative states. |
| Top-two oracle helps, but the learned planner does not capture that benefit | Repair prediction/ranking within supported options first. |
| All-six oracle adds meaningful benefit beyond the top-two oracle, but learned predictions cannot identify it | Better outside choices exist; validate or improve predictions before expanding control authority. |
| Predicted choices reliably capture outside benefit while meeting the collision constraint | Develop one conservative override policy, retaining direct control as fallback. |

Do not select a new override threshold merely to reproduce oracle choices in
these 80 spent development states. They are diagnostic and development evidence.

## Subsequent confirmation gate

Only after selecting and freezing an implementation, metric definitions,
episode budget, and collision noninferiority margin should a new held-out
comparison be run. Evaluate the frozen direct baseline concurrently on the same
held-out episodes; its old 65% development completion rate is not a universal
threshold. Require evidence that planning improves full-episode outcomes while
preserving acceptable collision behavior. A software `PASS`, more option
switching, or better training loss does not satisfy that milestone.
