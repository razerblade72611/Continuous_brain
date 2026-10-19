# Continuous Brain v0.20.1 — Goal Leverage / Consequence Calibration Diagnostic

## Why

v0.20.0 demonstrated a real but weak causal goal effect:

- 108 same-state goal interventions
- 11 changed high-level option choices
- pooled goal-swap rate 0.101852

When a swap occurred, predicted consequences usually moved in the intended
direction, but realized behavior did not cleanly separate:
- completion profile: completion 0.10, collisions 31.15
- safety profile: completion 0.15, collisions 41.30

The direct learned-option baseline remained much stronger:
- completion 0.55
- collisions 20.20

Memory should therefore NOT be connected to the goal register yet.

## Questions

This diagnostic separates three possible causes.

### 1. Selector candidate bottleneck

v0.20 searched only the top-2 options proposed by the learned selector.

At every real direct-option boundary, v0.20.1 evaluates ALL learned options and
asks:
- would completion vs safety choose different options if all options were
  available?
- did top-2 filtering suppress a goal-driven option change?

### 2. Goal-score leverage

For every option it decomposes:
- imagined reward;
- terminal option Q;
- predicted success;
- predicted collision;
- learned goal value.

It compares the largest completion-vs-safety score swing to the historical
reward+Q margin between the strongest options.

If the goal swing is usually smaller than the historical margin, the explicit
goal simply cannot influence ranking strongly enough.

### 3. Consequence calibration

The diagnostic runs autonomous direct-option trajectories.

For each executed option segment it compares:
- predicted collision score vs realized collision count;
- predicted success score vs realized segment completion.

Weak correlation means stronger goal weights would amplify a poorly calibrated
model rather than create useful planning.

## No training / no admission claim

CPU reference backend only.
No model parameter is modified.
v0.19 remains closed.
No generalization or scientific admission claim is made by this diagnostic.

The result determines the next architectural change:
- widen/retrain selector support;
- normalize/recalibrate goal scoring;
- improve world-model consequence heads;
- or a combination.

Only after goal-conditioned planning is behaviorally meaningful should admitted
memory be connected to the goal register.
