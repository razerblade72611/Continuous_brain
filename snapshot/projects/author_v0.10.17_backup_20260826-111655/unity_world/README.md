# AUTHOR v0.9.4 Discovery World

This editor-side builder clones `FloorPlan1_physics` into `FloorPlan1Discovery_physics`, converts both stock architectural doors into native openable `Door` simulator objects, and adds the North Archive and East Experiment rooms.

v0.9.4 keeps the v0.9.3 hinge and portal-clearance repairs and adds a maneuvering contract for the humanoid agents:

- all stools are removed from the cloned scene before NavMesh generation;
- each generated room has a 2.20 m x 2.20 m clear central turnaround square;
- each generated room has a 1.25 m clear doorway corridor;
- 0.35 m invisible perimeter keep-out bands prevent wall-hugging reachable positions;
- furniture is moved to the perimeter and narrow behind-furniture pockets are sealed;
- loose generated-room floor `Statue` and `Box` props are removed;
- the editor build fails if a reserved maneuvering zone intersects an enabled non-floor collider.

Run `apply_discovery_world_patch.py`, `verify_discovery_world_patch.py`, sync to the Windows Unity project, regenerate the scene, rebuild the Linux player, then run `scripts/smoke_test_discovery_world.py`.
