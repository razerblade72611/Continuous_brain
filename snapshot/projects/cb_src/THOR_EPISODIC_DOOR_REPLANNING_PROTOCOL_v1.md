# Continuous Brain v0.27.0 — Real 3-D Episodic Door Replanning Bridge

## Starting point

The following are now preserved results:

- v0.23.1: frozen two-item ordered episodic-memory TEST PASS.
- v0.24.0: episodic selection -> explicit goal -> autonomous temporal behavior
  causal integration PASS.
- v0.25.2: frozen consequence-planning/replanning TEST PASS in the new symbolic
  route family.
- v0.26.0: real `FloorPlan1Discovery_physics` embodiment admission PASS.

v0.26 directly verified:
- two real openable `Door` objects;
- authoritative `isOpen`;
- reachable-space expansion from 178 to 609 cells after doors opened;
- north/east Discovery extensions;
- live RGB and depth;
- pose-consistent camera motion;
- protected cognitive checkpoints unchanged.

The synthetic planning phase is closed.

## Purpose

v0.27 is the first bridge from the admitted cognitive mechanisms to physical
behavior in the real custom Discovery World.

No cognitive weights are trained.

The experiment asks:

Can a late episodic query select a real 3-D door episode, make that physical
door the current goal, execute a real route to it, notice that its state no
longer matches the predicted plan, and alter behavior to recover?

## Episodic key-value binding

The v0.23 reader was trained and TEST-qualified on canonical recalled-content
keys.  It was not trained to encode arbitrary AI2-THOR RGB directly.

v0.27 therefore does NOT pretend otherwise.

Each stored item is a key-value episode:

key:
- frozen admitted canonical recall128

payload:
- real simulator-authoritative Discovery door episode:
  - objectId
  - 3-D position
  - doorway outward direction
  - main-room approach point
  - far-side target point

Slot 0 uses a frozen canonical LEFT recall key.
Slot 1 uses a frozen canonical RIGHT recall key.

The real door payloads are ordered deterministically as the two Discovery exits.

The frozen v0.23 FIRST/SECOND attention selects the slot.  The same selected
slot supplies the bound real door payload.

Thus this stage validates a key-value episodic bridge.

It does NOT claim raw visual-to-memory writing.  That becomes a later milestone.

## Physical executor

The bridge uses real AI2-THOR primitives, not teleportation for task execution.

TeleportFull is used only to establish controlled episode start positions.

Task execution uses:
- RotateLeft / RotateRight
- MoveAhead
- OpenObject
- GetReachablePositions

Movement paths are computed on the real 0.25 m reachable grid.

The target door is opened during agent behavior with ordinary `OpenObject`
without `forceAction`.

`forceAction=True` is reserved for experimental setup and exogenous surprise
injection, not for the agent's open-door macro.

## Three physical conditions

For every start point and both late queries:

### OPEN

The selected door is open when the plan is formed.

Initial macro plan:

`NAVIGATE_TO_APPROACH -> CROSS_DOOR`

Required:
- physical navigation succeeds;
- physical crossing reaches the selected door's far side.

### KNOWN_CLOSED

The selected door is closed when the plan is formed.

Initial macro plan:

`NAVIGATE_TO_APPROACH -> OPEN_DOOR -> CROSS_DOOR`

Required:
- ordinary non-forced `OpenObject` succeeds;
- crossing reaches the selected room.

### SURPRISE_CLOSED

The selected door is open when the plan is formed.

Initial plan:

`NAVIGATE_TO_APPROACH -> CROSS_DOOR`

After the agent physically reaches the approach point, the experiment closes
the selected door as an exogenous world change.

The predicted door state is therefore `open`, while authoritative reality is
`closed`.

Replanning condition must:
1. detect the state mismatch;
2. invalidate the stale remaining plan;
3. insert `OPEN_DOOR`;
4. open the door through ordinary `OpenObject`;
5. recompute crossing from actual reachability;
6. physically cross.

## Plan-once causal control

Every SURPRISE_CLOSED episode has a paired plan-once run from the same start,
same memory query, and same initial open-door plan.

After the surprise close, plan-once deliberately ignores the mismatch and
executes the stale crossing path that was computed while the door was open.

It should fail to reach the far side.

This is the real-3D analogue of the v0.25 causal replanning control.

## Memory intervention

The bridge also swaps both episode slots before a FIRST/SECOND query.

The selected real-world door target must swap with the slot contents.

This demonstrates that the route goal follows episodic binding/order rather
than a hard-coded fixed door choice.

## Starts

Three deterministic main-room start positions are chosen from real closed-door
reachable cells, away from both exits.

Each of:
- FIRST
- SECOND

is executed from every start under:
- OPEN
- KNOWN_CLOSED
- SURPRISE_CLOSED

SURPRISE also receives the paired plan-once control.

## Frozen DEV gate

Memory / binding:
- exactly two real door episodes
- FIRST/SECOND correct physical target >= .99
- slot swap changes physical target >= .99

Embodiment:
- approach navigation success >= .90
- OPEN physical crossing success >= .90
- KNOWN_CLOSED success >= .90
- native non-forced OpenObject success >= .90
- overall physical crossing >= .90

Replanning:
- SURPRISE replanning success >= .90
- SURPRISE mismatch detection >= .90
- SURPRISE replanning trigger >= .90
- SURPRISE plan-once success <= .25
- replanning advantage over plan-once >= .65

Protected cognitive artifacts remain byte-identical.

## Scope of a PASS

A PASS supports this narrow real-3D integration claim:

frozen episodic query
 -> selected bound real-world door episode
 -> physical door goal
 -> real grid navigation
 -> cached expected door state
 -> exogenous real simulator state change
 -> mismatch detection
 -> stale-plan invalidation
 -> physical OpenObject recovery
 -> real doorway crossing

It does NOT yet establish:
- raw RGB/depth episodic encoding;
- a learned 3-D consequence model;
- learned 3-D high-level route planning;
- transfer of the symbolic v0.25 model weights into AI2-THOR;
- broad embodied intelligence;
- consciousness.

## After PASS

Freeze this bridge and advance two architectural pieces rather than adding more
door-script polish:

1. **3-D sensory episodic encoder**
   - encode an observed object/door/location episode from RGB/depth/pose into a
     learned key compatible with the episodic store;
   - remove the canonical LEFT/RIGHT key tags used by this bridge.

2. **learned 3-D consequence model**
   - learn temporal macro consequences from real Discovery transitions;
   - predict navigation/open/cross outcomes and state changes;
   - use prediction errors for replanning exactly as qualified in v0.25.

Once those are stable, reuse the already-trained AUTHOR door/navigation
competence as the motor layer instead of keeping this deterministic bridge.

## After FAIL

Do not modify the preserved cognitive milestones.

Classify the failure as:
- memory-to-real-payload binding;
- physical grid navigation;
- interaction pose / OpenObject;
- crossing geometry;
- mismatch injection;
- plan-once control construction.

Repair only the 3-D bridge.
