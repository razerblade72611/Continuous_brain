# Your existing v0.11.4 child on DirectML

The default v0.12 run has a separate small starter and state directory. These
commands operate on your existing trained child using the preserved v0.11
architecture and evaluators. They perform **no new training** and never advance
the failed growth run to 10%.

The source ZIP did not contain your trained checkpoint files. Supply the files
from your own stopped local state or downloaded frozen Drive endpoints.
`infant_training.pt` is a file; a directory path alone is not the checkpoint.

## Three-controller comparison

On Windows, double-click `LEGACY_DIAGNOSTICS.cmd` and enter the checkpoint's
path. In your WSL DirectML terminal:

```bash
bash run_local.sh legacy --checkpoint "/full/path/to/infant_training.pt"
```

This runs the same 100 spent seeds **12001–12100**, 120-step horizon, and three
controllers: reactive, direct options, hierarchical planning. Runtime, source,
protocol, and checkpoint hashes bind the results. Matching cached evaluations
can be reused; a device/version change invalidates that cache.

The historical direct-options result was **61/100**, hierarchical planning
**38/100**, and reactive **0/100**. The 61% result was not the reactive baseline.
`p4b_floor_passed` retains the original criterion for comparison; it does not
mean a new checkpoint has preserved 95% of the older capability.

## The prepared gate crossover

```bash
bash run_local.sh legacy \
  --micro "/full/path/to/frozen_micro.pt" \
  --low "/full/path/to/frozen_low.pt"
```

The supplied endpoints must match the recorded hashes:

| Endpoint | SHA-256 |
|---|---|
| Micro | `324bb845941926c4a6c2a30b9fd1c185784542445a782327837ffe6b614d8e98` |
| Low | `fa0142e29cbd80648b00629c4effcaaf122161dad21440e371c22caa106b9825` |

The diagnostic evaluates both checkpoints at 2% and 5%. Each new copy changes
only `planning.growth.gate` and `options.growth.gate`. All other tensors and the
source file are checked. The two gates move together, so this crossover does not
isolate the planner and option gates individually. Copies are explicitly
diagnostic artifacts and are not qualified training endpoints.

## Paired action trace

After identifying an episode where direct options succeeded and planning failed:

```bash
bash run_local.sh legacy \
  --checkpoint "/full/path/to/infant_training.pt" \
  --trace-seed 12001
```

This runs just that paired episode, records both action sequences, predictions,
actual outcomes, option decisions, and the first action divergence. The first
different action is a lead for causal investigation, not proof of the failure's
cause. Only the already-spent seed range 12001–12100 is accepted.

Outputs default to `~/ContinuousBrain/legacy_diagnostics` in WSL, or
`%USERPROFILE%\ContinuousBrain\legacy_diagnostics` on Windows. You can select a
different new output folder with `--output-dir`. The source checkpoints remain
unchanged. Reserved seeds **51001–51100** are never used by this workflow.

For the older training/data APIs, see `RUN_v0.11.4.md` and the preserved
`continuous_brain.parented_cli`. The v0.12 launcher does not schedule the old
growth stages or invent missing crossover/qualification results.
