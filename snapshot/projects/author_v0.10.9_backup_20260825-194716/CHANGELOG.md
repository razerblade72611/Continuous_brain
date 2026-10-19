# v0.10.9 — Runtime Stool Sanitizer

- Preserves all v0.10.8 maneuver-clear Discovery room geometry.
- Adds `DiscoveryRuntimeSanitizer.cs` under runtime `Assets/Scripts`.
- Attaches the sanitizer to the generated Discovery root.
- Runs stool cleanup in `Awake()` with `DefaultExecutionOrder(-32000)`.
- Removes runtime stools by `SimObjPhysics.Type`, object ID, asset ID, or GameObject name.
- Hardens editor removal with `Resources.FindObjectsOfTypeAll<SimObjPhysics>()`.
- Updates WSL patch installation, Windows Unity sync, and patch verification for the runtime sanitizer.
- Keeps stool absence as an authoritative runtime smoke-test requirement.
- Advances trainer/replay summary version to v0.10.9.
