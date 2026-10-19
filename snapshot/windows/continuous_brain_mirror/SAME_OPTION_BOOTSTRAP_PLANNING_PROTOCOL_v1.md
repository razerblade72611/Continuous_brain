# Continuous Brain v0.20.5 — Same-Option Bootstrap Planning

## Evidence motivating this repair

The current source exposes exactly the continuation primitive required:

`OptionCortex.action_value(latent, option, action_onehot)`

The historical `_policy_tail_value` instead chooses the best value among
selector-supported *next* options.

That is correct only after option termination.

v0.20.4 proved that enforcing the execution contract by open-loop rolling the
same option to predicted termination is also wrong in practice:

- historical hierarchical completion: 0.525
- continuation-consistent long-roll completion: 0.15
- historical collisions: 26.025
- long-roll collisions: 26.2
- illegal cap handoffs: 0

Therefore the one-step world model should not be forced to hallucinate tens or
hundreds of steps merely because real control remains committed.

## Repair

At a legal high-level decision boundary:

1. selector provides the same top-2 candidate options as the historical planner;
2. imagine each option for at most 12 primitive steps;
3. if learned termination occurs inside those 12 steps:
   - the search may branch to a new option;
   - a next-option policy tail can be used;
4. if the option is still active at step 12:
   - no branch is allowed;
   - compute the same option's own policy action;
   - bootstrap with
     `OptionCortex.action_value(latent, SAME_OPTION, policy_action)`.

So:

`short imagined reward + same-option learned continuation Q`

replaces both:
- the historical fake handoff at the 12-step cap; and
- v0.20.4's unreliable long open-loop rollout.

## Real execution is unchanged

The real controller still:
- commits to the selected option;
- applies that option's learned primitive policy;
- retains it until the learned termination head fires.

The planner is invoked again only at a genuine real option boundary.

## No training

No neural parameter changes.
CPU reference backend.
Memory preserved.
v0.18 intent preserved.
v0.19 remains closed.

## Spent-window stop rule

Seeds 14001-14040 are reused again because this is still diagnosis/repair.

Before running, the repair criterion is frozen:

- illegal cap handoffs = 0;
- same-option bootstrap must actually be used;
- completion >= 0.475;
- collisions <= historical hierarchical + 5;
- completion improves >= 0.20 absolute over v0.20.4's 0.15.

### If PASS

Do ONE frozen fresh-seed validation of the continuation-bootstrap planner.
If that validation holds, stop Detour planner repair and move immediately to
memory-guided autonomous planning.

### If FAIL

Stop this planner-repair loop.

Use the stronger learned direct-option controller as the execution backbone for
the next cognitive milestone, while keeping the planner/world model as an
experimental advisory mechanism until it is retrained on better long-horizon
targets.

This prevents continued benchmark gardening.
