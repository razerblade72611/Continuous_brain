# Continuous Brain v0.19.1 — planning/option audit replacement

This package supplies the missing integration audit. It is a new implementation
with `_v1_1` script names, distinct from the previously described but unavailable
`_v1` package. Earlier advertised hashes do not apply to these files.

The audit reads the project and checkpoints and writes only audit reports. It
does not import PyTorch or project modules, instantiate a brain, allocate model
weights, train, run an environment, or evaluate a held-out family. Python 3.10+
and the standard library are sufficient; the existing DirectML venv is fine.

## Install and run in WSL

Download `continuous_brain_v0191_planning_audit.zip` into Windows Downloads.

```bash
(
set -e
cd "$HOME/projects/continuous_brain_v0.12.2"
source "$HOME/.venvs/continuous_brain_v110_dml/bin/activate"

unzip -o \
  /mnt/c/Users/davei/Downloads/continuous_brain_v0191_planning_audit.zip \
  -d .

bash run_planning_option_integration_audit_v1_1.sh

cp state/planning_option_integration_audit_v1_1/audit_bundle.zip \
  /mnt/c/Users/davei/Downloads/continuous_brain_v0191_audit_bundle.zip
)
```

Upload `continuous_brain_v0191_audit_bundle.zip` for the integration review.
The subshell stops on a missing archive or failed command, without changing
the error-handling mode of your interactive shell.

The package adds uniquely named audit files. It does not contain replacements
for `continuous_brain/`, model weights, training scripts, or the project README.

## Output

The latest outputs are:

```text
state/planning_option_integration_audit_v1_1/report.json
state/planning_option_integration_audit_v1_1/report.log
state/planning_option_integration_audit_v1_1/audit_bundle.zip
```

Every invocation also retains its own output under `runs/<timestamp>/`.
An incomplete run replaces the latest outputs with its diagnostic report,
so an old successful report is not reused. Exit codes are 0 (audit complete),
2 (incomplete/inspection error), and 3 (output write failure).

If the audit reports INCOMPLETE, its diagnostic bundle is still useful. Copy it
with the `cp` command above and upload it together with the terminal output.

Expected completion messages include:

```text
READ_ONLY_AUDIT: true
TRAINING_PERFORMED: false
CHECKPOINT_MUTATED: false
CAPABILITY_EVALUATED: false
PLANNING_OPTION_INTEGRATION_AUDIT_COMPLETE
PLANNING_OPTION_INTEGRATION_AUDIT_RUN: PASS
```

PASS means the inspection completed and required source/base metadata were
available. It is not a planning, memory, or generalization capability pass.
Expected-parent hash mismatches and absent historical evidence remain explicit
findings even when inspection completes.

## What is collected

- AST-derived method signatures and source hashes for `PlanningCortex`,
  `OptionCortex`, `WorldModelEnsemble`, `CorticalNucleus`, `InfantBrain`, and
  relevant memory/intent wrappers; relevant source files are included.
- Checkpoint file inventory, stored tensor names/shapes/storage types, config,
  training counters and provenance. Missing planning tensors are reported as
  missing; no model loader initializes replacements.
- The four known current dependency paths and their expected hashes where known.
- Historical planning reports, with the original 100-episode P4b protocol and
  summary arithmetic checked against embedded episode records. A historical
  report is linked to a checkpoint only by an actual matching content hash.
  The hierarchical-versus-direct-option difference is reported separately.

No checkpoint weights, datasets, images, or virtual environments are bundled.
Metadata includes existing file paths and training provenance.

By default, the audit scans the current project plus sibling
`continuous_brain*` projects that have `state/` or
`continuous_brain_project/state/`. It skips datasets, assets, environments,
hidden directories, and directory symlinks. Extra locations can be included:

```bash
bash run_planning_option_integration_audit_v1_1.sh \
  --scan-root "$HOME/some_other_checkpoint_directory"
```

Metadata is inspected without allocating tensor storage. Full-file hashes are
computed before and after inspection for up to 12 prioritized checkpoints;
the known current parents are always included. Other discovered checkpoint
file sizes, modification times, and file identities are compared. Exact
content-hash coverage is listed in the report; un-hashed files are not
content-certified. Increase coverage with `--max-full-hashes 32` if needed.
Use `--current-project-only` to omit automatic sibling discovery.

PyTorch ZIP checkpoints containing ordinary dense tensors are supported. The
restricted pickle reader accepts only inert tensor/storage reconstruction
descriptors and limited standard containers. Unsupported custom globals,
legacy non-ZIP formats, absent storage members, and incomplete archives are
reported. It never falls back to executing pickle globals or project code.

## Validation and limits

The included standard-library tests cover checkpoint metadata parsing, absent
and corrupt storage, refusal to execute an unapproved pickle global, source
inspection without import, historical evidence completeness/arithmetic,
unchanged input bytes, detection of externally changed inputs, stale-output
replacement, runner exit propagation, and nested historical project discovery.
They use explicit synthetic fixtures, not trained neural models.

The metadata reader was also checked against a recovered real 14K-checkpoint
pickle stream: it identified planning, options, world, nucleus, and spatial
state and the recorded training counters without importing torch. This was
metadata-only validation. The local large checkpoint copy was incomplete, and
the full audit correctly returned INCOMPLETE for it. The user's intact local
checkpoints have not been inspected by this replacement yet.

To run the software tests yourself, optionally:

```bash
python -m unittest -v test_planning_option_integration_audit_v1_1.py
```

No model training, new timing head, held-out evaluation, or automatic checkpoint
promotion is part of this package.
