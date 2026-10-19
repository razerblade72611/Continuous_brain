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
