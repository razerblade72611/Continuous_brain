# Continuous Brain v0.30.2 — Joint-Self Goal Boundary Diagnostic

## Purpose

v0.30.0 and v0.30.1 both remain formal DEV FAILS.

The corrected v0.30.1 counterfactual self-consequence assay PASSED strongly:
- future-self-guided action accuracy = 1.00
- true self-blind world-only accuracy = 0.00
- donor-self action flip = 1.00

The only unresolved gate is future active-goal prediction:
- source joint model DEV = 0.9762
- repair head DEV = 0.9774
- required = 0.98

TRAIN and DEV are nearly identical, so another larger classifier is not
justified without understanding the error structure.

v0.30.2 is therefore diagnostic only.

## Questions

1. Does the exact frozen v0.28 EndogenousGoalArbiter reproduce the true next
   goals when given the TRUE next continuous-self state?

   It should. If not, the target construction is inconsistent.

2. When the same arbiter is applied to the PREDICTED next continuous-self
   fields, where does it fail?

3. Are learned-head errors concentrated near:
   - INVESTIGATE boundaries:
     - prediction_error = 0.55
     - uncertainty = 0.82
   - commitment A/B equality boundary
   - NONE / nonzero-commitment boundary?

4. Did v0.30.1 actually correct some errors while creating others?

## Important representational issue

The predicted continuous-self head uses sigmoids.

Therefore an exactly zero unresolved commitment is difficult to represent as
mathematical zero, while the frozen v0.28 arbiter defines NONE using a very
small zero test.

The diagnostic reports a range of NONE tolerances:
- 1e-6
- .005
- .01
- .02
- .05
- .10

These are diagnostics only.

No threshold is selected, admitted, or written into the architecture based on
this DEV result.

## Data

READ ONLY.

Reuses:
- v0.30 DEV 40000..44999

Does NOT touch:
- v0.30 TEST 60000..64999

No model is trained.

## Outcome

If errors cluster around arbitration boundaries:

Do not keep increasing classifier capacity.

The next family should carry active intention as explicit recurrent self-state
and train/evaluate multi-step self trajectories, so a tiny numerical error in a
continuous variable does not erase personal goal continuity.

If errors are not boundary-localized:

The joint latent is dropping goal-relevant information despite low continuous
self MAE. The next family should strengthen goal/intention representation
directly.

In either case, v0.30 TEST remains sealed because DEV never passed.
