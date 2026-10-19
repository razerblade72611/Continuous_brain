# Information contract — v0.6.2 (persistent core and motor lab)

| Role | Examples | Access and restrictions |
|---|---|---|
| Brain input | Egocentric RGB features, depth sectors, sensed beacon/waypoint, previous acknowledgement/reward; internal self-state, intention and retrieved sensor memories | Only the explicit `sensory`, `self`, `goal`, `memories` packet reaches the cognitive network. Unknown keys are rejected. No free-form simulator metadata is accepted. |
| Teacher signal | Planner-composed skill label, development reference motor, curriculum goal generation and shaped reward | Labelled in each experience/dashboard. Training loss and early teacher forcing use labels. Executive evaluation disables its teacher forcing and imitation updates. Motor-source labels remain visible. |
| Evaluator | Top-down camera, exact room coordinates, task-success checks, simulator diagnostics | Display/report only. Exact coordinates are not added to the cognitive packet. Ground-truth task outcomes may be used as reward/learning targets; they are not omniscient perception. |
| Safety supervisor | Primitive-action masks, sensor health, recovery budgets, stop requests | Overrides/rejects actions independently. Faults never become competence failures or memory vectors. This is simulator protection, not a certified physical safety system. |

## Known legacy privilege, explicitly retained

The THOR adapter is the existing v0.5.3 implementation. It still bootstraps pose
belief from simulator metadata, uses simulator task generation and some action
masking, and may store privileged diagnostic information in the archived old
memory. Internal-map planning removes a route oracle; it does not remove all
simulator privilege. This release is not robot-ready and does not claim otherwise.

Legacy memories therefore remain an intact, provenance-labelled archive and are
not automatically retrieved by the new executive. The new development room's
beacon is an explicitly simulated line-of-sight bearing/range sensor. A physical
implementation would need an actual perception/localization equivalent.

Neither a larger parameter count, nonzero recurrent activations, the individual
UUID, nor its use of words such as "self" establishes subjective experience.
No aversive emotional states or autonomous code modification are implemented.

## v0.6.2 motor additions

- Controller input: the existing local six-value sensor/reward interface and
  requested orient/navigate mode. Sensor normalization does not read environment
  metadata. The reward scalar is the existing training feedback channel, not an
  independently certified physical sensor. The new motor does not use RGB features.
- Teacher: the reference controller labels demonstrated local actions in the motor
  lab. Labels are used for supervised gradients; they are absent from qualified
  motor action selection. Motor-lab training never loads the persistent individual.
- Evaluator/curriculum: world coordinates may set initial poses, verify a clear
  corridor and score actual movement. Neither the policy nor its feature encoder
  receives those coordinates. Qualification tests local motor skills, not general
  embodied competence or the learned executive.
- Supervisor: valid primitive masks and qualification/body-contract checks. Masks
  do not force forward actions. This remains simulator control, not robot safety
  certification. The learned motor has no hidden teacher fallback.
- System records: motor digests, qualification reports, replacement archives and
  controller-update counts are provenance/training controls, not omniscient neural
  sensory input. Old failures remain history; run-specific metrics distinguish them.

`--evaluate --planner-diagnostic` is explicitly an exception to executive-teacher
removal: it uses the authored planner for diagnosis while disabling all training
and persistent writes. Plain `--evaluate` retains the teacher-free executive mode.
