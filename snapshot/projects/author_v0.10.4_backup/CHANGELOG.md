# v0.10.4 — Pose-Authoritative Short-Range Navigation

- Fixes Level-1 false negatives where `NavigateTo` reached a simulator-returned interaction pose but the outer verifier rejected it using object-transform-center distance.
- `ThorExecution` now carries explicit controller `goal_verification` evidence for pseudo-action goals.
- Pose-aware `NavigateTo` uses exact X/Z pose proximity, yaw, horizon, visibility, and simulator action success as authority; object-center distance is diagnostic only.
- Staged curriculum Levels 1–3 pin the exact `GetInteractablePoses` pose used during episode construction, preventing near-approach lessons from silently retargeting a different pose.
- Adds explicit reasons for pose-goal completion/failure (`interaction_pose_reached`, replan exhaustion, pinned pose unavailable, blocked-door stage failure, step-budget exhaustion).
- Preserves v0.10.3 spatial actionability, bounded interaction recovery, A/B checkpoints, replay, and SFT history.
- Adds v0.10.4 regression coverage for far-from-transform valid poses, authoritative failure override, compatibility fallback, and curriculum pose pinning.

# v0.10.3 — Spatial Actionability + Reliable Recovery

- Generalizes exact interaction outcomes into X/Z position-level actionability.
- Migrates v0.10.2 exact-pose history into spatial rows automatically.
- Excludes failed moving-object X/Z cells during recovery.
- Prefers spatially different, safe-clearance candidates before route cost.
- Adds explicit recovery attempts, successes, exhausted, and unclassified legacy counts.
- Uses actionability recovery when opening route-infrastructure doors.
- Adds position-level inspection output and v0.10.3 replay schema.
- Preserves worker-seed delta merging for exact and spatial actionability.
- Adds v0.10.3 regression coverage for spatial migration, generalization, recovery selection, and skill-schema migration.

# v0.10.2 — Learned Interaction Actionability + Recovery

- Adds persistent per-agent exact-pose actionability learning for physical interactions.
- Treats `GetInteractablePoses` as visibility/range candidates rather than guaranteed action-success poses.
- Adds bounded alternate-pose retries for open/close/pickup/place/toggle while retaining the same semantic target.
- Adds a soft open/close clearance prior plus learned exact-pose penalties and rewards.
- Records internal interaction attempts, error messages, recovery status, pose keys, and distance telemetry.
- Seeds worker actionability from persistent A/B history and delta-merges multi-worker updates without duplicating old counts.
- Marks recovered runtime/training skill successes with recovery statistics.
- Adds `scripts/inspect_actionability.py`.
- Fixes current-run coverage metrics so persistent map history cannot produce >100% run coverage.
- Adds v0.10.2 actionability/recovery regression tests.

# v0.10.1 — Verified Interaction-Pose Curriculum

- Replaces object-center interaction navigation with `GetInteractablePoses` and connected path-cost pose selection.
- Adds bounded pose replanning and preserves primitive-step accounting across failed replans.
- Adds controller-private, bounded door-aware route continuation for a previously selected semantic target.
- Splits body metrics into staged approach/interaction skills and records auxiliary door stages encountered inside routes.
- Migrates v0.10 checkpoints losslessly under `legacy_v010`; legacy navigation is not trusted as staged approach competence.
- Makes invalid setup unscored and trainer cleanup authoritative; removes `close_cleanup` as a learned skill.
- Adds graduated curriculum levels and `LEVEL=` support in the WSL training wrapper.
- Exports only successful verified motor demonstrations to SFT JSONL.
- Hardens peer text transport with message IDs and fail-closed sentinel/malformed-JSON filtering.
- Adds live-session updates to each agent's own staged body-skill checkpoint.
- Adds v0.10.1 interaction-pose preflight tooling and regression tests.

# v0.10.0 — Accelerated Embodied Skill Curriculum

- Rewrites the accelerated trainer around a weighted curriculum: navigation, door-open-and-cross, pickup, place, toggle, and peer meeting.
- Adds persistent independent A/B body-skill checkpoints under `state/checkpoints/a|b/embodied_skills.json`.
- Adds `CrossThreshold`, a body-only pseudo-action that crosses the same open doorway already selected by the LLM; it never chooses a semantic target.
- Normal task mode can use trained Navigate/Face/Cross macros after the model has grounded the requested target.
- Keeps direct verified interactions available even before training; learned skills improve the precondition and crossing mechanics.
- Adds `state/replay/embodied_replay.jsonl` with simulator-authoritative action/result transitions.
- Adds `state/datasets/embodied_sft.jsonl` containing successful target-type/action-sequence examples suitable for later SFT/LoRA preparation.
- Training writes compact per-agent competence summaries into that agent's own persistent episodic memory.
- Keeps shared map/local-navigation learning and separate private A/B memory.
- Default training makes zero LLM calls.
- Adds curriculum selection via `--curriculum` and `--no-sft-export`.
- Adds skill inspection utility.

# v0.9.3 — Physical Doorway Repair + Anti-Echo Peer Text

- Rebuilds Discovery doors around a true hinge/swing child so the leaf collider actually swings out of the doorway instead of rotating in place around its own center.
- Clears runtime static flags from the moving door hierarchy.
- Adds build-time doorway-prism collision clearance for stock decorative wall / DoorFrame colliders, then restores invisible boundary collision on both sides of each opening.
- Keeps the closed door leaf as the physical blocker; opening it exposes a real corridor with no teleport.
- Keeps v0.9.2 exact `Floor` walkable threshold/floor naming.
- Enhances the Discovery smoke test to report post-open `isOpen` state and perform a physics command boundary before reachability.
- Adds anti-echo peer text and preserves explicit user-requested text delivery.
