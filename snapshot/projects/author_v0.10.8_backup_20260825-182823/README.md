# AUTHOR v0.10.8 — Maneuver-Clear Discovery Rooms

v0.10.8 preserves the v0.10.7 selected-event fix and v0.10.6 simulator watchdog while hardening the custom Discovery World geometry for Level-3 navigation.

The Unity Discovery builder now removes stools, reserves 2.2 m x 2.2 m turnaround zones, reserves 1.25 m doorway corridors, adds wall keep-out buffers, moves furniture to room perimeters, removes loose generated-room floor clutter, and validates the maneuvering zones before saving the scene.

Because this release changes Unity scene generation, the custom `FloorPlan1Discovery_physics` player must be regenerated and rebuilt after upgrading AUTHOR.

See `AUTHOR_v0.10.8_OPERATOR_README.md` for the exact upgrade/build/test sequence.
