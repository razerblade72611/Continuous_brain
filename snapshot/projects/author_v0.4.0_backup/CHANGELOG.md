# Changelog

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
