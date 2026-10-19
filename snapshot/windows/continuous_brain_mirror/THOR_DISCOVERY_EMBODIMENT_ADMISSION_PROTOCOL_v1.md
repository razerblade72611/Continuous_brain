# Continuous Brain v0.26.0 — Discovery World Embodiment Admission

## Why this exists

v0.25.2 is a frozen TEST PASS for the new consequence-planning/replanning
mechanism.

The synthetic cognitive phase is now closed for this stage.  Before connecting
those mechanisms to real 3-D control, the exact local Discovery World runtime
must be verified rather than assumed.

v0.26.0 performs no learning and changes no cognitive checkpoint.

It launches the user's real custom AI2-THOR executable:

`~/projects/ai2thor-humanoid/builds/thor-discovery-Linux64/thor-discovery-Linux64`

with scene:

`FloorPlan1Discovery_physics`

## Checks

The admission verifies:

- the requested Discovery scene actually loads;
- at least two real `Door` objects exist;
- at least two doors report `openable=true`;
- `OpenObject` changes simulator-authoritative `isOpen` state;
- `GetReachablePositions` returns substantial walkable space;
- both north and east Discovery extensions are reachable;
- open-world extents reach at least 5m in both +X and +Z;
- first-person RGB exists;
- RGB changes after a real camera rotation;
- simulator pose yaw changes with rotation;
- yaw returns after the inverse rotation.

Depth presence is reported but is not a gate in this bridge admission, because
the next cognitive integration can initially operate on authoritative
topological/door state while the richer visual/depth representation is brought
in separately.

Stool objects are reported.  Their presence is descriptive rather than a gate.

## Dependency handling

The wrapper first tries the active Continuous Brain venv.

If that environment cannot import `ai2thor`, it automatically uses:

`~/.venvs/author/bin/python`

provided that venv can import `ai2thor`.

This does not copy AUTHOR neural state into Continuous Brain.  It uses the
known-working simulator controller dependency only.

## Preservation

The following are hashed before and after the simulator admission:

- canonical InfantBrain checkpoint;
- admitted working-memory checkpoint;
- v0.23 episodic reader;
- remembered-goal decoder;
- v0.25 consequence model;
- v0.25.2 frozen TEST result.

Any change invalidates the run.

The simulator is reset and stopped before exit.

## Meaning of PASS

A PASS establishes that the current real Discovery World runtime exposes the
physical interfaces required for the next integration stage.

It does not establish that the v0.25 symbolic consequence model transfers to
3-D without adaptation.

## After PASS

Build the first real 3-D cognitive integration around a deliberately narrow
task:

1. observe and remember two exit/door episodes;
2. receive a late goal/query;
3. select the remembered target;
4. construct a topological route to the selected door/region;
5. execute through the existing embodied motor boundary;
6. verify predicted door/route state after each temporal action;
7. inject or encounter a door-state mismatch;
8. invalidate the stale plan and replan.

The first 3-D milestone should be an integration/development experiment, not a
claim of broad embodied intelligence.

## After FAIL

Do not alter cognitive checkpoints.

Classify the issue as:
- missing AI2-THOR Python controller;
- executable launch failure;
- scene/build mismatch;
- door metadata/action mismatch;
- reachability regression;
- sensor/pose regression.

Repair only the simulator/bridge layer.
