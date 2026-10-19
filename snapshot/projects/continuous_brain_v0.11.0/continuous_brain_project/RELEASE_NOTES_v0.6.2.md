# v0.6.2 — Train, qualify and attach a separate development motor

## Why this release exists

The user's controlled v0.6.1 test produced zero navigation advances with the
imported motor, versus 457 with the reference motor under the same authored
planner. Motor history survived, but compatible tensors did not imply behavioral
transfer. This release preserves the motor/executive separation while no longer
requiring the frozen THOR motor to work unchanged in another body/environment.

## Implemented

- Fresh 5,060-parameter feed-forward motor with explicit orient/navigate conditioning.
  Nine normalized input features are derived from the existing six sensors plus
  two-way skill mode. Goal direction/distance normalization is deterministic sensor
  preprocessing, not a turning rule. No reference controller is called at inference.
- Four learned primitive outputs: idle, advance, left and right. Interaction is
  excluded from this motor's contract. External masks restrict orientation to idle/
  turns, but do not force forward movement or substitute teacher actions.
- Independent motor lab: actual development-room observations labelled by the
  reference controller, class-balanced imitation learning, persistent optimizer,
  dataset/RNG, nonfinite checks, bounded checkpoints, CPU execution and clean stop.
- Separate teacher-free qualification using a distinct seeded distribution of
  visible goals/clear corridors. Candidate reports are bound to exact tensor and
  interface digests; stale, insufficient or failing qualifications cannot attach.
- Non-overwriting attachment to a continuation copy. Preserve the persistent
  executive/optimizer/target/replay/workspace/memories/identity/environment/RNG.
  Archive the replaced motor and retain all old legacy files. Checkpoints write
  version 0.6.2, and compatible v0.6.0/v0.6.1 checkpoints remain readable.
- Motor dynamics qualify only for DevelopmentRoom. No silent THOR/robot transfer.
- New-controller replay is partitioned as before. World-model consequence reranking
  now warms up on updates for the newly attached motor, not lifetime updates for
  a different controller. No executive parameter or optimizer reset is performed.
- Built-in, read-only `--evaluate --planner-diagnostic` mode, run-specific task
  counts and primitive counts by skill; no lengthy diagnostic monkey-patching needed.
- Backups now include archived replacement motors. Older upgrade utility retains
  v0.6.2 identity/version and archives rather than silently downgrading a checkpoint.

## Scope and non-claims

This is supervised acquisition of elementary steering skills. It is not autonomous
skill invention, comprehensive navigation, semantic vision/language, a literal
biological motor cortex or evidence of consciousness. The motor is trained in a
separate lab and held fixed during executive runs, not jointly adapted online.
Its initial demonstrations are fixed; repeated training refines them rather than
automatically harvesting the individual's lifetime motor experience.

The qualified component is narrower than the whole system. Success in clear
corridors does not guarantee success on arbitrary room tasks. Overall evaluation
still includes authored search/inspection and a bounded goal scheduler. The normal
executive training run also retains a decaying planner teacher. Native Unity,
physical embodiment and pretrained semantic foundation integration remain outside
this patch; no additional simulator-recovery work was introduced.
