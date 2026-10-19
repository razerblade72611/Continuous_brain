# Continuous Brain — Competitive Broadcast v2

This package finishes the competitive-workspace experiment against the real
`InfantBrain` in the supplied Continuous Brain project.  It is an additive
research branch: it does **not** replace or rewrite the learned base brain and it
does not modify the source checkpoint.

## What changed from the previous draft

The earlier draft had four integration problems that made a result hard to
interpret:

1. the evaluator's baseline was a newly initialized `InfantBrain`, not the same
   frozen trained base used by the experimental conditions;
2. checkpoint loading ignored the saved growth configuration and vision-parent
   head dimensions;
3. the `values` branch was called a trained consumer even though the task had no
   valid care-vector target (the real values head predicts care state, not an RL
   scalar);
4. competition was downstream of the recurrent core, so the new mechanism could
   not itself learn to retain the cue.

v2 fixes all four.  The saved InfantBrain is loaded strictly, including growth
and parent heads; it stays frozen and in eval mode (so BatchNorm buffers cannot
drift).  A zero-initialized competitive pathway now broadcasts to three real
consumers: recurrent hidden state, motor policy, and world-model workspace.  The
care/value head stays untouched.  Disabling competition always routes through
the exact original path.

Training uses the project's parented-development style: an optimal demonstrator
provides action labels while the learner sees only ordinary observations.  At
the delayed junction the label depends on a cue that has already disappeared,
so the network still has to retain the cue recurrently.  Teacher forcing removes
exploration coverage as a confound.  A simultaneous world-context prediction
loss gives the world consumer its own dense training signal.

## Files

Copy these into an existing project root (the ZIP is already laid out this way):

- `continuous_brain/competitive_infant_brain.py`
- `continuous_brain/cued_detour_world.py`
- `continuous_brain/evaluate_competitive_infant.py`
- `continuous_brain/test_competitive_infant.py`
- `continuous_brain/test_cued_detour_world.py`
- `scripts/run_competitive_broadcast_experiment.sh`

No existing core source file is overwritten by this patch.

## Validate before training

From the project root, with the project's Python environment active:

```bash
python -m continuous_brain.test_cued_detour_world
python -m continuous_brain.test_competitive_infant
```

Expected final lines are:

```text
All checks passed for real, in this sandbox, against the actual environment.py base class.
All competitive-broadcast checks passed under real PyTorch.
```

The competitive test checks: exact no-op immediately after growth, exact fallback
after perturbing only the trainable branch, exact save/restore continuation,
frozen/eval base state, expected parameter accounting, and non-zero gradients to
all three independent selectors once the zero-init residuals have moved off zero.

## Run the experiment

The supplied project archive contained this checkpoint:

```text
state/child_v113_dml_smoke/infant_training.pt
```

It has 14,000 recorded updates and loads strictly with `balanced-v110` growth and
the saved 384- and 768-dimensional vision-parent heads.  If you have a better
qualified child checkpoint, use it instead; all three conditions **must use the
same exact checkpoint**.

One-command run on Linux/WSL/ROCm (PyTorch ROCm uses the `cuda` device name):

```bash
DEVICE=cuda \
BASE_CHECKPOINT=state/child_v113_dml_smoke/infant_training.pt \
UPDATES=4000 \
./scripts/run_competitive_broadcast_experiment.sh
```

CPU smoke/full run:

```bash
DEVICE=cpu ./scripts/run_competitive_broadcast_experiment.sh
```

Windows DirectML, using the module commands directly, can pass `--device directml`.

The script trains three matched conditions with the same training seed:

- `shared`: one selector broadcasts to recurrent state, motor, and world model;
- `independent`: three separately parameterized selectors;
- `uniform`: shared branch with selector query frozen at zero, so weights stay
  exactly 1/4 while residual capacity can still learn.

It then evaluates the frozen base, shared, independent, uniform, hidden-reset,
and competition-disabled conditions on paired seeds and writes:

```text
state/competitive_broadcast_v2/competitive_broadcast_results.json
```

## Reading the result

`baseline_disabled_exact_parity` must be `true`; otherwise the experiment is
invalid because the fallback path failed to preserve the original brain.

The primary structural test is `shared_minus_independent`.  Positive success-rate
difference favors forced common broadcast; overlap with zero is inconclusive;
a negative result counts against the specific shared-selection hypothesis.  The
independent model has more selector parameters, so a shared win is conservative,
while an independent win cannot by itself distinguish structural benefit from
extra capacity.

`shared_minus_hidden_reset` asks whether persistent recurrent state is actually
load-bearing.  `shared_minus_uniform` asks whether learned stream selection adds
anything beyond the residual adapters themselves.  World-context MSE is reported
alongside behavior so the world consumer cannot be declared successful merely
because it exists in the architecture.

These are functional/mechanistic tests.  Passing them would justify integrating
the competitive broadcast branch as a protected optional growth stage in the
main development pipeline; it would not by itself establish subjective
experience or consciousness.

## Validation already performed on the supplied archive

The package was executed under real PyTorch against the supplied source.  The
environment/no-leak suite passed.  The full preservation/gradient suite passed.
The supplied 14,000-update checkpoint loaded with `strict=True`; zero-initialized
growth reproduced its hidden state and motor logits exactly; and a real
teacher-supervised optimizer step completed successfully.  A tiny three-condition
smoke evaluation also verified exact baseline/disabled parity.  Those smoke runs
are integration checks only, not the scientific result of the 4,000-update run.
