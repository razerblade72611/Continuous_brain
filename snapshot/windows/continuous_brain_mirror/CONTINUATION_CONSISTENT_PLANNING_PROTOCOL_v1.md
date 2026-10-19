# Continuous Brain v0.20.4 — Continuation-Consistent Planning Diagnostic

## Established mismatch

The reproduced continuation experiment confirms that historical
`planning_v103` treats `max_option_steps=12` as both:
- a computation limit, and
- an effective macro boundary for higher-level search.

Real execution does not.

If learned termination remains false, the active option retains control.

In the reproduced completion-loss cases, options imagined for 12 un-terminated
steps actually remained active for 103–120 real steps.

Oracle continuation comparisons also establish that useful initial choices
exist:
- best-realized top-2: 5 rescued failures, 0 broken successes;
- best-realized all-6: 10 rescued failures, 0 broken successes.

Those oracle rows are diagnostic only because they use future information.

## Repair

No learned tensor changes.

`12` becomes a compute chunk only.

At a chunk boundary:
- keep the same option;
- keep rolling the learned planning latent;
- keep applying that option's learned primitive policy;
- keep checking its learned termination head.

A new option may enter the beam only if:
1. learned termination is predicted; and
2. real episode time remains.

If no termination is predicted before the remaining episode horizon, the same
option owns the entire imagined remainder.

No next-option Q tail is added when no handoff is legal.

## Why this run reuses 14001–14040

This is a repair diagnostic, not a promotion gate.

The window has already been spent repeatedly during v0.20.1/v0.20.2 and the
external continuation study. Reusing it lets us determine whether correcting
one semantic contract actually helps before consuming another development
window.

## Comparison

The run reports:
- direct learned options;
- historical capped hierarchical planner;
- continuation-consistent hierarchical planner.

The selector remains top-2.
No goal term is used.
No collision-goal term is used.
Memory remains disconnected from planning goals.
v0.19 stays closed.

If the repaired planner improves the spent-window autonomous behavior without
regression, the next step is a separately frozen fresh-seed admission test.

If it does not, the next target is world-model long-horizon fidelity /
termination prediction, not memory or goal weighting.
