# Changelog

## 0.3.1

- Added `--step-delay` to slow each primitive AI2-THOR movement/rotation/interact action.
- Added `simulation.step_delay_seconds` configuration (default 0.15 s).
- Applied delay inside navigation paths, not just between LLM decisions.
- Improved llama.cpp compatibility: retry a 400 request once without `response_format`.
- LLM errors now include the server response body for useful diagnostics.

## 0.2.0

- Fixed composite pickup-and-place goals being marked complete after pickup only.
- Added a dedicated goal parser and goal verifier.
- Strengthened `PutObject` verification to require the source object's `parentReceptacles` to contain the requested target receptacle.
- Added deterministic `NavigateTo` using AI2-THOR `GetReachablePositions`, BFS grid routing, cardinal turns, and primitive `MoveAhead` actions.
- Added navigation-aware recovery instead of blind movement retries.
- Added LLM planner validation for supported actions and real object IDs.
- Added separate primitive-action and replanning budgets.
- Updated WSL2/RX 6600 documentation and helper scripts.
- Expanded the test suite from 3 tests to 13 tests.
- Cleaned generated caches, run logs, and packaging artifacts from the distributable archive.

## 0.3.0 - Free-roam autonomy

- Added `--roam` mode that bypasses goal-completion checks entirely.
- Added continuous LLM observe -> decide -> act -> recover loop.
- Added optional Qwen2-VL camera-frame input with `--vision`.
- Added `--interact` for autonomous pickup/open/close/place behavior while roaming.
- Added `--roam-decisions`, `--vision-every`, and `--max-steps` runtime controls.
- Added deterministic fallback exploration if the LLM endpoint or JSON response fails.
- Added per-decision console output and `roam_decision` episodic-memory records.
