# Continuous Brain v0.27.1 — Runtime Dependency Bridge Repair

## Classification of the v0.27.0 stop

v0.27.0 did not reach its real 3-D experiment.

The observed sequence was:

- package hashes matched;
- Python compilation passed;
- `THOR_EPISODIC_DOOR_REPLANNING_SMOKE: PASS`;
- wrapper then stopped with:
  `ERROR: no usable Python environment with ai2thor + torch.`

This is an infrastructure/dependency-boundary stop, not a scientific FAIL.

No v0.27 episode was executed and no 3-D DEV result was consumed.

## Cause

The active Continuous Brain venv contains the project's PyTorch stack but does
not expose `ai2thor`.

The AUTHOR venv contains the known-working `ai2thor` controller used to launch
the custom Discovery executable, but the original v0.27 wrapper required that
same interpreter to import PyTorch too.

That requirement was unnecessary.

## Repair

v0.27.1 does not install, upgrade, copy or modify either venv.

It keeps the active Continuous Brain Python interpreter and appends the AUTHOR
venv's `site-packages` directory to `sys.path` **after** the active
interpreter's existing paths.

Consequences:

- Continuous Brain's torch remains authoritative.
- Continuous Brain's numpy and other installed dependencies remain
  authoritative.
- `ai2thor` is resolved from AUTHOR as a fallback.
- dependencies missing in Continuous Brain can fall back to AUTHOR.
- no package installation is required.

The original v0.27 evaluator and cognitive source are unchanged.

## Scientific status

This patch changes only runtime import composition.

It changes:
- no cognitive weights;
- no environment;
- no door logic;
- no starts;
- no thresholds;
- no DEV conditions;
- no causal controls.

Therefore the first successful v0.27 experiment remains the first scientific
3-D bridge run.

## Stop rule

If the merged import probe succeeds and the evaluator runs:
- interpret the original v0.27 gate normally.

If the merged import probe fails:
- do not install arbitrary package versions yet;
- inspect the missing import/dependency reported by the bridge and repair only
  that runtime dependency.
