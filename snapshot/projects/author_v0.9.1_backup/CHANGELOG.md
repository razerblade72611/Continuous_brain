# v0.9.1 — Verified Task Execution

- Replaced claim-based social task completion with evidence-based task contracts.
- Explicit peer text tasks require an actual `TextPeer` action and a delivered `TextMessage` to the intended recipient after task assignment.
- Unsupported/free-form `task_complete=true` claims are logged as ignored and no longer return the body to autonomous mode.
- Added user-task-only motor handoff from repeated preparation to `PickupObject`, `PutObject`, `OpenObject`, `CloseObject`, `ToggleObjectOn/Off`, and `DropObject` once the model-selected focus is within reach and advertises the requested affordance.
- Added composite pickup/place staging without exposing hidden object IDs to the LLM.
- Added physical Discovery-region verification for `Escape`/`Exit` tasks; crossing a real doorway is required.
- Added door-opening handoff for exit tasks only after the model itself has selected/grounded a reachable door.
- Added `task_mechanical_contract` to task-mode prompts; it is derived solely from the current user instruction and contains no hidden scene catalog or coordinates.
- Added persistent Unity compiler fix by fully qualifying `UnityEngine.SceneManagement.SceneManager.MoveGameObjectToScene`.
- Expanded regression coverage from 106 to 115 tests.

# v0.9.0 — Two-Room Discovery World

- Adds a Unity editor builder that clones FloorPlan1 into `FloorPlan1Discovery_physics`.
- Converts both architectural kitchen doors into native AI2-THOR `CanOpen` objects.
- Adds a North Archive/Signal Room and East Experiment/Workshop Room.
- Adds readable language displays plus stock manipulable AI2-THOR objects.
- Rebuilds the NavMesh so agents physically walk through opened doors; no teleport.
- Adds a Discovery World smoke test for both door reachability branches.
- Training/session helper scripts default to the Discovery scene/player and remain overrideable.
- Text tasks now require an actual delivered TextPeer event instead of trusting `task_complete`.
- Active manipulation tasks explicitly discourage repeated Approach once the target is in reach.
- Retains v0.8.0 grounded manipulation, texting, persistence, accelerated training, and v0.6.1 humanoid patch.

# v0.8.0

- Added explicit distance-independent TextPeer transport with persistent transcript.
- Added grounded ToggleObjectOn/Off and DropObject capabilities.
- Extended object metadata normalization and verifier for toggled/drop state.
- Added live interaction smoke test.
- Fixed training coverage metric to never exceed 100%.
- Retained v0.7 accelerated learning and v0.6.1 humanoid Unity patch.

# Changelog

## 0.7.0 — accelerated learning + persistent skills

- Added `python -m agent_core.training` / `thor-train`, a no-LLM accelerated trainer for low-level navigation experience.
- Added clean `--reset-memory` state reset before training and a guarded standalone reset utility.
- Added persistent cross-run episodic/social memory with safe LLM recall that excludes old camera prose and simulator object IDs.
- Added an experience-derived persistent topological room map.
- Added a lightweight tabular local navigation policy with JSON checkpoints.
- Added a persistent navigation replay buffer.
- Added randomized A/B spawn positions and headings for training episodes.
- Added explicit peer-encounter training episodes without injecting a normal-mode social preference.
- Added optional multi-process AI2-THOR workers and deterministic worker-state aggregation.
- Added training coverage, success, peer-encounter, step-throughput, and history metrics.
- Normal semantic `Approach` can use the learned map/checkpoint and safely falls back to deterministic AI2-THOR navigation when coverage is missing.
- Added `learning:` and `training:` configuration blocks plus `scripts/run_training_wsl.sh`.
- Added regression tests for memory persistence/privacy, reset safety, map routing, checkpoint persistence/merge, learned motor-layer routing, and training dry-run behavior.

# v0.6.1 — Humanoid Visual Replacement Hotfix

- Confirmed the v0.6.0 Unity hook executed for both agents; the remaining defect was visual attachment/replacement.
- Kept AI2-THOR's native `TallVisCap` as `VisibilityCapsule` rather than assigning a sibling visual root.
- Permanently suppresses stock peer renderers with `Renderer.forceRenderingOff` so later `IsVisible` toggles cannot resurrect the old body.
- Re-centered humanoid geometry around AI2-THOR's centered 1.8 m tall controller (`~ -0.9 .. +0.85 m` local Y).
- Added a more explicit humanoid silhouette: pelvis, torso/chest, neck/head/face/eyes, shoulders, upper/lower arms, hands, thighs, knees, shins, and feet.
- Reuses the stock visible layer, disables primitive colliders, disables shadow casting, and re-registers the native visibility host with first-person culling.
- Added in-place upgrade support from the v0.6.0 patch with `.v061.bak` rollback files.
- Added regression tests for TallVisCap hosting, persistent stock suppression, centered coordinates, v0.6.0 upgrade, and rollback.
- Python social/LLM/session behavior is unchanged from v0.6.0.

# v0.6.0 — Humanoid Social Embodiment

- Added explicit stable self identity (`Agent A` / `Agent B`).
- Added explicit peer ontology: the other label denotes another independently embodied agent like the model itself.
- Peer camera-cone telemetry now carries the known peer label rather than anonymous body geometry.
- Kept peer private camera, memory, task, focus, intention, and hidden reasoning isolated.
- Added `observation` and `intention` output channels and stopped replaying verbose observation/status prose into later model prompts.
- Added compact persistent self-chosen intention state.
- Compact mechanical corrections no longer masquerade as visual perception.
- Observer Dashboard now displays current observation, self-chosen intention, and observer status separately.
- Added `ThorConfig.local_executable_path` and CLI `--thor-executable` for custom AI2-THOR players.
- Added `unity_humanoid/HumanoidPeerAvatar.cs`, a runtime humanoid peer renderer that does not alter the stock physics body.
- Added a safe AI2-THOR source patcher with backup/revert support and verification tooling.
- Added regression tests for context hygiene, explicit peer identity, custom executable routing, and Unity patch application/revert.

# Changelog

## 0.5.2 — peer perception telemetry + stagnation/JSON mechanics

- Explicitly initialize shared AI2-THOR with `makeAgentsVisible=True`.
- Added neutral peer camera-cone geometry sensing with distance and signed bearing.
- Peer sensor reports camera-cone membership, not guaranteed unoccluded pixel visibility.
- Added dashboard peer distance/bearing/camera-cone/hearing telemetry and in-view-cone badge.
- Added observer-only `peer/report mismatch` flag for camera-cone presence without peer-like model wording.
- Added neutral recent-progress proprioception and repeated-action/stagnation diagnostics.
- Added conservative response repair for trailing text, one-line action intents, and plain no-op reports.
- Preserved model-authored visual reports even when they disagree with peer telemetry.
- Preserved social neutrality: no greeting/cooperation/approach preference is injected.
- Added regression tests for peer geometry, prompt privacy, dashboard telemetry, response repair, and visible-agent initialization.

## 0.5.1 — observer dashboard

- Added a read-only dual-camera browser dashboard for two-agent social sessions.
- Shows Agent A and Agent B RGB viewports side-by-side from the same current shared-world event.
- Shows each agent's explicit vision/status report, selected focus, chosen semantic actions, task/mode, pace, and body feedback.
- Shows explicit speech/hearing and a shared observer event log.
- Highlights LLM fallback/parse/grounding failures without changing agent behavior.
- Dashboard receives copies of runtime output only after decisions; nothing from the observer is fed back into either LLM context.
- Hidden chain-of-thought, private memory, system prompts, simulator object catalogs, and the other agent's private state are not exposed.
- Added `--dashboard`, `--dashboard-host`, `--dashboard-port`, and `--dashboard-refresh-ms`.
- Added `observer:` configuration block and `scripts/run_social_dashboard_wsl.sh`.
- Uses only Python's standard-library HTTP server; no new web-server dependency is required.
- Added observer privacy, dual-frame, fallback-visibility, and read-only HTTP regression tests.

## 0.5.0

- Added native shared-scene AI2-THOR multi-agent support using `agentCount=2`.
- Added per-agent adapters that route every Unity action with explicit `agentId`.
- Added independent per-agent camera frames, step counters, pacing, memories, semantic focus, and task state.
- Added `Speak` as a non-Unity social action distinct from human-facing `say`.
- Added range-limited speech bus; only explicit spoken text crosses agent boundaries.
- Added serialized A/B inference against one shared local Qwen server/model weights.
- Added `/a`, `/b`, and `/both` task/roam command routing.
- Added separate log directories under `runs/agent_a` and `runs/agent_b`.
- Added `scripts/smoke_test_multi_thor.py` and `scripts/run_social_wsl.sh`.
- Preserved v0.4.1 metadata-neutral perception/grounding and embodiment mechanics.
- Added social/multi-agent privacy and routing regression tests.

## 0.4.1

- Added neutral proprioceptive feedback after every embodied decision.
- Added persistent model-selected semantic focus and post-selection affordance state.
- Added semantic `Face` action.
- `Observe`, `Wait`, and empty action lists are valid choices.
- Primitive action targets are treated as attention only, never hidden steering.
- Repeated `Approach` is rejected once approach distance is already satisfied.
- Enabled internal instance segmentation and viewport-ID grounding; IDs remain hidden from the LLM.
- Added regression tests for neutrality, viewport grounding, semantic focus, proprioception, and idle behavior.

## 0.4.0 — neutral embodiment + persistent task session

- Removed the AI2-THOR object candidate catalog from autonomous LLM prompts.
- Added `grounding.py`, a hidden semantic-target -> AI2-THOR object ID translation layer.
- New semantic targets must resolve to currently visible objects; unseen metadata cannot turn hallucinated objects into actions.
- Removed controller-generated wandering fallback. On LLM failure, the body idles with `Pass`.
- Added `session.py` for one persistent scene that switches between autonomous and user-task modes.
- Added `--session` and `--initial-task` CLI options.
- Plain terminal text assigns/replaces the current user task; `/roam` returns to autonomy.
- Structured pickup/place/open/close tasks use hidden simulator state for completion verification only.
- Unstructured tasks can be completed by the model with `task_complete=true`.
- Reworked `vision.py` to expose the camera plus minimal body state while excluding object metadata and raw simulator errors.
- Preserved model-selected `slow`, `normal`, and `fast` movement pace.
- Added privacy-boundary tests proving autonomous prompts do not contain simulator object names/IDs.
- Added grounding tests proving hidden objects cannot be selected through omniscient metadata.
- Added persistent-session command/state tests.
- Preserved the v0.2 deterministic goal/verifier/navigation stack for legacy one-shot task mode.

## 0.3.2

- Added short LLM-controlled movement bursts and model-selected pacing.
- Added compact candidate aliases and target cooldown.

## 0.3.1

- Added per-primitive-action movement delay and improved llama.cpp 400 diagnostics.

## 0.3.0

- Added continuous free-roam mode, optional camera input, and autonomous interactions.

## 0.2.0

- Fixed composite goal semantics and exact PutObject verification.
- Added deterministic NavigateTo, navigation-aware recovery, and separate action/replan budgets.
