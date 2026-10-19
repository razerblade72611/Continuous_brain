# Continuous Brain v0.11.4

Build `v114-protected-integration-r2` implements protected growth after the reported
v0.11.0 planning regression. Use the included Colab notebook for the complete run.

## Run in Colab

1. Download `continuous_brain_v0.11.4.zip`, its matching `.zip.sha256`, and
   `Continuous_Brain_v0.11.4_PROTECTED_GROWTH_COLAB.ipynb`.
2. Open the notebook in Colab and select a GPU runtime. A100 40 GB is the intended
   runtime; the notebook records whichever GPU is actually assigned.
3. Choose **Runtime → Run all**. Mount Drive when Colab requests it. If the upload
   picker appears, select the release ZIP and checksum from Windows Downloads.
4. Existing v0.11.0 inputs are reused from `MyDrive/continuous_brain_v110/`:
   `continuous_brain_v110_colab_bundle.tar.gz`,
   `v110_world_retention_assets.tar.gz`, and its `.sha256`.
   A missing input triggers an upload picker. For the large bundle, uploading it
   through Drive's web interface first is also supported.
5. Let the notebook run baseline, protected stages, and qualification. A P4b
   failure stops the next stage. Reconnecting and using Run all resumes from the
   last verified checkpoint without adding a second allocation of updates.

The ZIP contains the full source project and notebook. It does **not** contain
trained weights or nursery datasets; those come from your existing v0.11.0 bundle.

## Source and progression

The branch must start from exact-expansion SHA-256:

`086932388a9a410af46fea8e87cc63dd4b60161e03e247bf81fc88d28cfd577f`

The failed 6000-step Phase A checkpoint is retained as experimental history and
is never used as this branch's source.

| Stage | Planning / option visibility | Target updates |
|---|---:|---:|
| shadow | 0% | 2,000 |
| micro | 2% | 1,000 |
| low | 5% | 1,000 |
| moderate | 10% | 1,000 |

These are scheduled, non-trainable gates. The model cannot open them itself.
Shadow learning uses an explicit auxiliary forward of the grown planner/options;
normal inference keeps those residuals closed. After each stage, a matching P4b
result on spent seeds 12001–12100 must pass before the next stage starts.

The final sequence is bootstrap retention, sealed Foraging/Detour world retention,
and randomized-detour development on 41001–41100, requiring completion ≥20%.
Reserved blind seeds 51001–51100 are not evaluated by this workflow.

## Saved outputs

Your protected branch is:

`MyDrive/continuous_brain_v114/state/child_v114_protected_r2/`

It includes `infant_training.pt`, a previous periodic checkpoint, measured
training receipts, and a frozen endpoint for each completed stage. Periodic
checkpoints are copied and verified every 250 updates. A disconnect can require
repeating up to 249 unsaved updates. Drive filesystem persistence is subject to
Colab/Drive availability; a failed transfer stops training and reports the error.

Results and streamed logs are:

`MyDrive/continuous_brain_v114/results/protected_integration_r2/`

Main outputs: `V114_FINAL_SUMMARY.json` and `V114_PROTECTED_READINESS.json`.
If a gate stops the notebook, send its stage JSON and `.stdout.log` for diagnosis.

The separate `r2` branch avoids mixing this corrected protocol with any earlier
v0.11.4 experimental branch. Resume is exact in the tested CPU environment;
bit-identical continuation across different GPUs or PyTorch versions is not
promised. Qualification caches are invalidated when the recorded runtime changes.

## What this release changes

- Protects existing growth with fixed stage gates, preserving learned capacity.
- Rehearses successful original Detour training trajectories alongside new layouts.
- Uses the frozen old selector to retain option identities, because old checkpoint
  metadata did not retain the original clustering centers. It does not recluster
  the anonymous option IDs on new layouts.
- Anchors planner state, transitions, selector/action distributions, termination,
  and Q ordering using old Detour training data. Current planner latents feed the
  option comparison, so compatibility covers the planner-to-option interface.
- Applies general perception/workspace/language/control anchors on every route;
  the previous modulo-five pattern reached only planning/options.
- Enforces legacy tensor fingerprints, strictly loads protected checkpoints, checks
  input provenance, prevents stage overrun and reopening of earlier stages, and
  binds evaluation results to checkpoint, protocol, source manifest, and runtime.
- Copies periodic state to Drive without deleting the state directory or losing
  frozen stage endpoints. Readiness never invents missing success evidence.

General non-planning growth remains visible at 1.0. This is an experimental
protocol choice, not a guarantee of retention. Final retention gates remain required.

## Optional WSL source installation

The full Colab path above needs no WSL installation. To inspect/test the source
locally, use a new directory and your existing environment:

```bash
cd /mnt/c/Users/davei/Downloads
sha256sum -c continuous_brain_v0.11.4.zip.sha256
mkdir -p "$HOME/projects/continuous_brain_v0.11.4_r2"
unzip continuous_brain_v0.11.4.zip -d "$HOME/projects/continuous_brain_v0.11.4_r2"
cd "$HOME/projects/continuous_brain_v0.11.4_r2/continuous_brain_project"
python -m pip install -e . --no-deps --no-build-isolation
python -m continuous_brain.parented_cli train-protected-growth --help
```

Use your existing compatible PyTorch installation; local DirectML training
performance and GPU execution were not exercised by this release's CPU tests.
See `RUN_v0.11.4.md` for the expanded CLI protocol.

## Validation scope

See `RELEASE_VALIDATION_v114.md` for tests actually run for this build. Software
validation does not establish recovery from 13% to 38% or generalized completion
≥20%. Your trained checkpoint must produce that evidence in Colab.
