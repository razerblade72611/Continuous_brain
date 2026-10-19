# Continuous Brain — Matched Recall Phase 1 v1

Status: **authorized after both recall arms passed production smoke**

DEV and TEST remain sealed.

## Experiment

Train M1.5 (`linear`) and normalized M2 (`gated`) for exactly one complete
1600-condition TRAIN epoch from separate fresh copies of the untouched canonical
14k checkpoint.

Both use:
- identical condition order;
- one exact LEFT/RIGHT counterfactual pair per optimizer update;
- LR `1e-4`;
- DMLSafeAdam;
- schedule seed `73191`;
- init seed `12345`;
- motor objective:
  `0.50 * mean-step CE + 0.50 * delayed-decision CE`;
- world-loss weight `0.25`;
- save every 100 updates;
- full historical-base verification every 50 updates.

No arm initializes from the 25-update smoke checkpoint.

## Resume integrity

The production trainer is identical in architecture/objective to the smoke
trainer but adds checkpoint-consistent JSONL reconciliation.

If interrupted after the last saved checkpoint:
- the entire existing JSONL is preserved as a timestamped sibling backup;
- only records after the checkpoint update are removed from the active JSONL;
- retained records must be exactly updates `1..checkpoint_update`;
- retained arm labels and condition IDs must reproduce the deterministic
  schedule exactly;
- training then resumes from the portable checkpoint.

Duplicate existing update numbers are treated as an error requiring manual
inspection.

## Train-side audit

After both 1600-update arms finish, the runner verifies:
- exactly 1600 rows per arm;
- exact train-set coverage once per arm;
- identical condition order;
- identical realized decision timing;
- all writer/recall gradients present at their expected updates;
- workspace exact every update;
- all 50-update whole-base verifications present.

It reports separately:
- five-action full correctness;
- lateral-action-top-choice rate;
- LEFT-vs-RIGHT side-sign correctness;
- pair correctness for all three;
- side margins;
- realized-hidden-delay slices.

If the existing M1 writer-only training JSONL is present, its side-sign metrics
are included as a historical train-side baseline.  M1 cannot supply the newer
full-action/lateral-top metrics because that older production log did not record
them.

## Interpretation discipline

This full TRAIN epoch is still not DEV evidence.

We will not pick a winner from aggregate accuracy alone.  After this run:
1. inspect full-action vs side-sign behavior by realized delay;
2. compare both arms against M1's side-sign failure;
3. run train-side causal recall interventions on any arm that materially repairs
   the long-delay failure;
4. only then freeze the updated DEV evaluator and open DEV.

No DEV/TEST access is authorized by this runner.
