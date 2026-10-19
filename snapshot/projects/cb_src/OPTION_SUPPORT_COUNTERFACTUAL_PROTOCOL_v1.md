# Continuous Brain v0.20.3 — Option Support Counterfactual Diagnostic

v0.20.2 falsified the naive "remove selector filtering" repair.

On seeds 14001-14040:

- direct learned-option baseline:
  - completion 0.65
  - collisions 20.7

- all-option neutral planner:
  - completion 0.175
  - collisions 69.3

- all-option success-goal planner:
  - completion 0.15
  - collisions 71.975

- success vs neutral option-swap rate:
  - 0.01

Therefore the selector is not merely an obstacle to goal control. It appears to
carry information about which temporal options are competent in the current
state.

This read-only diagnostic reuses the already-spent 14001-14040 window.

At up to 80 real direct-option decision boundaries it:
1. records every option's selector logit/rank;
2. reconstructs the exact DetourWorld boundary by deterministic action replay;
3. executes every learned option counterfactually for one real option segment;
4. records realized completion, collision delta, reward and segment length;
5. compares top-2 vs outside-top2 realized competence.

No training occurs. No new development window is consumed. v0.19 remains closed.

Interpretation:
- If top-2 options are more competent but useful options sometimes exist
  outside top-2, use selector support as a soft competence prior.
- If outside-top2 options are almost always poor, retain hard selector support
  and improve goal sensitivity only inside supported options.
- If selector support does not predict realized quality, investigate option
  execution/model-based ranking instead of preserving the selector by
  assumption.
