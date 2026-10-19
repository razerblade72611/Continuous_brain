# Continuous Brain v0.35.1 — Endogenous Goal Completion Dynamics Diagnostic

## Status

v0.35.0 remains a formal DEV FAIL.

Its only failed gate was:

`self_generated_goal_completes_from_self_estimate`

Observed:
- completion within 8 target episodes = 0.46875
- required = 0.80

Everything upstream and downstream of that gate was strong:

- autobiography-generated target matches the fragile commitment = 1.00
- donor autobiography flips the self-generated target = 1.00
- erased autobiography generates no directional target = 1.00
- active goal survives unrelated interruption = 1.00
- self-generated goal controls the target plan = 1.00
- mean target capability gain = +0.55688
- discrepancy reduction rate = 1.00
- mean late/bootstrap discrepancy ratio = 0.05475
- lifecycle persistence = 1.00

TEST 100000..100127 remains sealed.

## Diagnostic question

Why does the autobiographical completion criterion fail in more than half the
DEV subjects despite large capability improvement and near-elimination of
expected-vs-actual discrepancy?

Three broad explanations are possible:

1. **Horizon problem**
   The +0.12 autobiographical reliability target is coherent, but eight
   target-directed episodes are too few.

2. **Self-estimate saturation / criterion mismatch**
   Capability and discrepancy improve, but the frozen autobiographical
   reliability signal does not increase by +0.12 for many subjects under this
   repeated single-target developmental trajectory.

3. **Insufficient development**
   The failed subjects genuinely do not improve enough, so goal completion
   should remain low.

## Method

No training.

Reuse only spent v0.35 DEV subjects:
- 90000..90127

Keep TEST sealed:
- 100000..100127

For each subject:

1. reconstruct the exact bootstrap autobiography;
2. generate the exact self-authored v0.35 target;
3. preserve the exact unrelated interruption;
4. continue SCAFFOLD development on the self-authored target for 32 episodes;
5. read frozen autobiographical reliability after every episode;
6. record the first episode at which baseline +0.12 is reached;
7. separately record capability gain and autobiographical discrepancy.

Report completion at:
- 8
- 16
- 24
- 32 episodes

Also report:
- interruption-induced reliability shift;
- peak and final reliability gain at 8 and 32;
- fraction never reaching +0.12 by 32;
- discrepancy reduction at each horizon;
- capability gain at each horizon;
- developmental state of the original eight-episode noncompleters.

## Interpretation

If completion reaches >=.80 by 16:
- the v0.35.0 mechanism likely has an assay-horizon problem.

If it reaches >=.80 only by 32:
- the same interpretation holds, but the self-estimate is slow relative to
  capability development.

If completion remains <.80 by 32 while the original noncompleters still show
strong capability and discrepancy improvement:
- the fixed +0.12 reliability target is not a good completion variable for this
  endogenous-goal trajectory;
- do not lower the threshold against DEV;
- instead redesign self-authored completion around the quantity the goal is
  actually trying to change, with a new preregistered DEV repair or new family.

If capability/discrepancy also fail:
- do not reinterpret completion;
- the developmental behavior itself is inadequate.

## Integrity

This diagnostic:
- does not train;
- does not alter source checkpoints;
- does not touch v0.35 TEST;
- cannot convert v0.35.0 from FAIL to PASS.

Its sole purpose is diagnosis before any repair.
