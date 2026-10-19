# AUTHOR v0.9.3 Discovery World

This editor-side builder clones `FloorPlan1_physics` into `FloorPlan1Discovery_physics`, converts both stock architectural doors into native openable `Door` simulator objects, and adds the North Archive and East Experiment rooms.

v0.9.3 keeps the exact `Floor` walkable collider children introduced in v0.9.2, but also repairs the door mechanics: each leaf is parented under a true hinge/swing transform, runtime static flags are cleared, and the doorway prism is cleared of decorative stock collision. Invisible boundary segments are rebuilt around the portal so the opening is physically traversable only where the door is located.

Run `apply_discovery_world_patch.py`, `verify_discovery_world_patch.py`, sync to the Windows Unity project, regenerate the scene, rebuild the Linux player, then run `scripts/smoke_test_discovery_world.py`.
