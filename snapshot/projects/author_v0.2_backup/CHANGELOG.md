# Changelog

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
