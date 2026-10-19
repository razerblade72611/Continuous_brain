# Continuous Brain — Working Memory Recall Production Smoke v1

Status: **authorized only after Recall Qualification v2 passed**

DEV and TEST remain sealed.

## Purpose

This is a 25-update execution of the exact production trainer that will later be
used for the 1600-condition arm comparison.

It is not an alternate toy trainer.

## Matched-arm contract

M1.5 (`linear`) and M2 (`gated`) use identical:

- canonical 14k base checkpoint;
- writer architecture;
- initial seed `12345`;
- condition schedule seed `73191`;
- first 25 structural train conditions;
- exact LEFT/RIGHT pair per optimizer update;
- DMLSafeAdam with LR `1e-4`;
- 50/50 mean-motor vs delayed-decision motor objective;
- world-loss weight `0.25`;
- base-preservation checks;
- portable CPU checkpoint format.

Only the recall module differs.

## Behavioral reporting

Every delayed decision records separately:

1. `full_correct`: whether the five-action argmax equals the teacher side;
2. `lateral_top_choice`: whether LEFT or RIGHT is even the five-action argmax;
3. `side_correct`: whether RIGHT-vs-LEFT latent sign favors the correct side;
4. signed correct-side margin.

This avoids hiding the frozen-substrate failure mode in which the full argmax
can be `down` while the latent side preference remains strongly RIGHT-skewed.

## Production-smoke gates

Each arm must independently:

- run all 25 updates;
- stay entirely within TRAIN;
- keep candidate gradient positive;
- keep writer gate gradient positive after update 1;
- keep direct recall projection gradient positive after update 1;
- for M2, keep recall-gate weight and bias gradients positive from update 3;
- preserve workspace exactly after every optimizer step;
- preserve whole historical base at updates 5/10/15/20/25;
- finish with the exact historical base digest;
- reproduce exact historical behavior when both writer and direct recall are
  disabled;
- write a CPU-portable checkpoint;
- report no DirectML CPU fallback.

## Cross-arm gates

The smoke comparison additionally requires:

- identical update numbers;
- identical condition IDs for all 25 updates;
- identical delayed-decision timing;
- numerical equality through update 3 for the neutral shared forward path
  (losses, memory magnitude, recall magnitude, writer gradients, projection
  gradients) within fixed floating tolerance.

After update 3 the arms are allowed to diverge because M2's contextual gate has
just become trainable.

## Interpretation

The 25-update smoke is still an implementation gate, not evidence that either
arm solves long-delay memory.

No full 1600-update training begins unless both arm smokes and the cross-arm
comparison pass.
