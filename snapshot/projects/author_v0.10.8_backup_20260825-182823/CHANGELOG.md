# v0.10.8 — Maneuver-Clear Discovery Rooms

- Updates `DiscoveryWorldBuilder` to v0.9.4.
- Removes all stools from the cloned FloorPlan1 Discovery scene before NavMesh generation.
- Adds build-time validation that no stool-like SimObj/GameObject remains.
- Reserves a 2.20 m x 2.20 m obstacle-free turnaround square in both generated rooms.
- Reserves a 1.25 m-wide clear doorway-to-center corridor in both generated rooms.
- Adds 0.35 m invisible keep-out bands along the three non-door walls of each generated room.
- Moves Archive and Experiment furniture to the perimeter.
- Removes loose generated-room floor `Statue` and `Box` props that could become rollover/trap obstacles.
- Seals narrow behind-furniture pockets from reachable navigation space.
- Adds build-time collider intersection checks for both turnaround zones and both doorway corridors.
- Extends `smoke_test_discovery_world.py` to verify stool absence and maneuver-zone reachability.
- Preserves v0.10.7 selected-event-safe trainer behavior and v0.10.6 simulator watchdog behavior.
- Advances embodied replay/SFT metadata and training summary version to v0.10.8.
