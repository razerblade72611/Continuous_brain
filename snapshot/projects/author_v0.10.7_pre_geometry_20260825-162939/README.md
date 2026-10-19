# AUTHOR v0.10.7 — Selected-Event-Safe Watchdog Curriculum

v0.10.7 fixes the agent-B trainer event-selection regression discovered by the Level-3 watchdog test. `SharedThorWorld.step_for()` already returns the requested agent Event; trainer helpers now consume that Event directly instead of indexing its `.events` collection a second time.

The v0.10.6 FIFO watchdog, unscored timeout handling, Unity restart, persistent A/B state isolation, pose-authoritative navigation, and actionability recovery remain intact.

See `AUTHOR_v0.10.7_OPERATOR_README.md`.
