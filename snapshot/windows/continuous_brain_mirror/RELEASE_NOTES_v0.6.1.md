# v0.6.1 — Legacy motor continuity and useful search

## Architecture purpose

Keep the persistent executive independent from replaceable, source-labelled motor
controllers. Preserve learned state while correcting an authored option that made
some hidden targets unreachable from the available action hierarchy. This is a
bounded v0.6 correction, not another Unity navigation redesign.

## Changes

- Recognize the reported v0.4.3 format-6 embodied curriculum schema without a scene
  marker. Unknown/foraging schemas remain rejected. Validate all tensor keys/shapes,
  finite weights, visual dimensions and recorded visual-backbone identity.
- Add non-overwriting, locked-source copy-on-upgrade. Optional motor attachment
  preserves the current executive identity, weights, optimizer, workspace, memories,
  replay and physical state. Its motor recurrence begins fresh. Original files
  remain unchanged. Interrupted copies are marked and cannot run.
- Keep legacy mastery as recorded provenance rather than silently recertifying it
  in a different environment. Freeze imported motor parameters.
- Search retains scan/relocation counters across option boundaries and checkpoints,
  uses center clearance, depth validity and action acknowledgements, and can move
  without first seeing the target. No new skill output dimension is required.
- Partition replay by controller revision. Retain historical rows but do not fit
  current-world outcomes to obsolete action semantics or bridge a controller change
  in a recurrent training sequence. No executive weight/optimizer reset.
- Correct per-experience motor provenance: imported neural motor applies only to
  motor-controlled skills, not authored search, idle or interaction options.
- Preserve v0.6.0 checkpoint loading; write v0.6.1 checkpoints. Keep all v0.5.3
  legacy entry points and sensor behavior unchanged.

## Information contract for these changes

| Category | Information / role |
|---|---|
| Brain/controller input | Existing six sensors and image features; search uses only sensor values and its own persisted counters |
| Teacher signal | Existing authored planner and optional development reference motor, both explicitly labelled |
| Evaluator | Development target/world coordinates used only by environment/evaluation; never passed to search |
| Safety supervisor | Sensor-health validity and primitive action mask; unhealthy depth cannot authorize search translation |

Checkpoint provenance, upgrade receipts and controller-revision labels are system
records. They do not inject simulator ground truth into neural input. Legacy
episodic databases remain archived and unclassified rather than being relabelled
as semantic memories. The existing THOR coordinate/odometry assumptions have not
been removed by this patch; see the base architecture contract.

## Limitations

Search is a bounded heuristic, not complete exploration or learned skill discovery.
Tests show relocation and target reacquisition in selected cases, not universal
success. Older motor transfer may regress because imagery/dynamics differ. The
executive remains an early research system; more training alone does not supply
missing semantic, language, social or open-ended learning capabilities.

Use RUN_v0.6.1.md for current instructions. Earlier release documents are retained
as historical context; their old import instructions are superseded by this release.
