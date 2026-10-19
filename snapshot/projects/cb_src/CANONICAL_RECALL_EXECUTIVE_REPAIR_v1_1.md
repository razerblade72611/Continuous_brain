# Canonical Recalled-Content Executive v1.1 Repair

Scope: production trainer field-name repair only.

Observed failure:
`KeyError: 'executive_residual'`

Cause:
The v0.14.0 canonical wrapper intentionally exposes the new executive output as
`executive_delta`, but two lines in the production trainer still referenced the
v0.13.2 field name `executive_residual`.

Repair:
- replace exactly two `out["executive_residual"]` references with
  `out["executive_delta"]`.

Unchanged:
- architecture;
- admitted checkpoint and SHA;
- loss;
- optimizer;
- learning rate;
- schedule seed;
- initialization seed;
- TRAIN/DEV/TEST definitions;
- fresh family;
- admission thresholds;
- evaluator;
- decision gate.

The failed run stopped before the first production optimizer update completed,
so no production checkpoint needs recovery. Re-running the existing
`run_canonical_recall_executive_v1.sh` after installing this replacement is the
correct continuation.

Fresh DEV and TEST were not touched by the failed production-training attempt.
