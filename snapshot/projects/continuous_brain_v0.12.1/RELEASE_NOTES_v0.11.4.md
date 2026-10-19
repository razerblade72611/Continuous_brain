# v0.11.4 release notes

Build: `v114-protected-integration-r2`.

Implements controlled integration of the existing 53.68M-class expanded model:
staged residual visibility, explicit auxiliary shadow learning, frozen-child
behavior/interface distillation, and successful original Detour rehearsal.

The source retains v0.11.1–v0.11.3 DirectML compatibility. It adds no learned
parameters. Protected branches require a new migration; this corrected protocol
does not silently resume an older experimental v0.11.4 branch.

Corrections to the recovered v0.11.4 package:

- Replace surrogate zero-valued shadow gradients with a measured auxiliary forward.
- Preserve inherited option IDs using the frozen selector; do not recluster new
  layouts into arbitrary IDs expected by existing heads.
- Anchor old Detour interfaces directly and feed current planner latents into
  option compatibility. Apply general anchors to every route.
- Enforce prior P4b evidence, exact update ceilings, optimizer/protocol continuity,
  strict checkpoint loading, and migration-time legacy fingerprints.
- Make repeated stage execution a no-op at its target; never reopen earlier stages.
- Mirror periodic checkpoints with verified temporary copies, preserving stage
  endpoints and the previous active checkpoint.
- Bind cached evaluations to checkpoint, runtime, source manifest, and settings.
- Remove fabricated integrity-result reconstruction and reject missing evidence.
- Accept the valid FP32 representation of a 10% gate using a numerical tolerance.
- Include a complete upload-capable Colab notebook and source/checksum validation.

Full run instructions are in `START_HERE_v0.11.4.md`. Training and capability
qualification of the user's actual child remain to be run in Colab.
