# Continuous Brain — Executive Composition Production v1

## Purpose

The 50-update Phase-1 smoke proved that the late-bound executive is connected,
trainable, initially neutral, and can be added without modifying the
DEV-admitted memory brain.

It did NOT yet learn INVERT.

This run therefore performs the first real executive learning experiment.

## Protected substrate

Frozen throughout:
- historical 14k brain;
- DEV-admitted 64-D writer;
- DEV-admitted nonlinear memory→workspace reader;
- frozen motor/world model.

Only the 24,960-parameter late-bound executive is optimized.

## Training unit

Each of 1600 optimizer updates uses exactly one structural TRAIN condition and
all four counterfactual members:

- LEFT + NORMAL
- RIGHT + NORMAL
- LEFT + INVERT
- RIGHT + INVERT

Every TRAIN structural condition is used once under a deterministic shuffled
schedule.

LR: `2e-4`
schedule seed: `92731`
init seed: `24680`

Loss:
- 50% mean post-instruction step CE
- 50% first delayed lateral-decision CE
- averaged over all four counterfactual members.

## Fixed-final causal gate

After update 1600, evaluate all 1600 TRAIN structural conditions at fixed
weights.

Pre-registered requirements:

- NORMAL pair full >= 0.85
- INVERT pair full >= 0.85
- four-way full >= 0.80
- NORMAL pair side >= 0.98
- INVERT pair side >= 0.98
- four-way side >= 0.95
- instruction flip consistency >= 0.98
- memory-swap donor consistency >= 0.98 for NORMAL and INVERT
- executive-off NORMAL pair side >= 0.95
- executive-off INVERT pair side <= 0.10
- memory-zero INVERT pair side <= 0.10
- NONE instruction exact no-op <= 1e-6
- every factor level INVERT pair side >= 0.90
- every factor level INVERT pair full >= 0.60

These are TRAIN learning criteria only.

## Interpretation

PASS:
- the brain-integrated executive has learned to use the same stored content
  differently under a late-arriving goal;
- both memory and instruction are causally necessary;
- old admitted cognition remains protected.

Then stop tuning on this family and construct a fresh second-generation
executive structural family with new frozen TRAIN/DEV/TEST splits.

FAIL:
- do not open or construct an executive DEV evaluation yet;
- inspect whether failure is optimization, scale, or architecture;
- preserve the admitted memory brain unchanged.

## Held-out policy

The already-opened working-memory DEV split is not reused for executive
development.

The old TEST split remains sealed.
