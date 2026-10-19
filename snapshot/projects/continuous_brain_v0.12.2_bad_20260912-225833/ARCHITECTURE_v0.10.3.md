# Continuous Brain v0.10.3 Architecture

## Hierarchical Option-Level Imagination

v0.10.3 is an integration release. It adds **no neural parameters** and performs **no migration of learned weights**. A trained v0.10.2 checkpoint is consumed bit-for-bit unchanged.

The change is architectural at runtime: primitive control and high-level imagination are assigned to different learned systems.

## Why this release exists

Development evidence from v0.10.2 established a sharp separation:

- the learned direct option controller completed 5/10 detours;
- restricting imagination to the option policy's single preferred primitive action preserved 5/10;
- allowing recurrent imagination to choose between the option policy's top two or top three primitive actions collapsed completion to 0/10.

That showed that useful temporal options had been learned, but unrestricted primitive-action planning was overriding them destructively.

## v0.10.3 control contract

### While an option is active

The active option owns primitive control:

```
current planning latent
        |
        v
active learned option
        |
        v
options.action_policy
        |
        v
primitive action
```

No primitive-action beam search is performed during this period.

### At a learned option boundary

The learned termination head decides when a high-level decision is permitted. The planner then searches over selector-supported option IDs rather than raw actions:

```
real planning state
      |
      v
learned option selector
      |
      +--> candidate option A -- deterministic option-policy rollout --+
      +--> candidate option B -- deterministic option-policy rollout --+--> compare
      +--> ...                                                        --+
                                                                         |
                                                                         v
                                                               chosen next option
```

Each candidate option is imagined as a **macro-action**. During its imagined rollout, every primitive action comes from that option's own learned primitive policy. The recurrent planning cortex predicts the resulting latent state and reward. The macro rollout stops when the option's learned termination signal fires after minimum commitment, or when `max_option_steps` is reached.

The high-level beam can continue for multiple option macro-actions (`option_depth`).

## Candidate filtering

Only the top `selector_topk` learned option IDs are expanded at each high-level boundary. Selector logits are used for candidate filtering and as an exact-score tie-break only; there is no arbitrary weighted blend between selector confidence and imagined reward.

## High-level score

A branch receives:

1. discounted rewards predicted during each option-policy macro rollout; plus
2. a terminal option-conditioned action value, evaluated only on the primitive action selected by selector-supported option policies.

There is no unrestricted `max` over all primitive actions at the tail.

## Primary development configuration

The release predeclares this configuration for the next `9001-9010` development test:

- `option_depth = 2`
- `selector_topk = 2`
- `beam_width = 8`
- `max_option_steps = 12`
- `discount = 0.95`
- `min_commit = 3`
- `terminate_threshold = 0.50`

The untouched final blind seeds remain `12001-12100`.

## Qualification gate

The formal P4b gate is unchanged:

- hierarchical completion >= 20%;
- completion advantage over reactive > 5 percentage points;
- collision regression <= 2 collisions per episode.

Direct option behavior remains a diagnostic baseline. It does not substitute for the designated hierarchical imagination result.

## Scientific scope

A pass would demonstrate learned hierarchical delayed planning in the fixed-gap DetourWorld regime. It would not by itself establish generalized planning across randomized layouts, AGI, consciousness, or personhood. Randomized-layout planning remains a later milestone.
