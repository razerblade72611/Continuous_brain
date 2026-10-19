# AUTHOR v0.10.26 — ExitRegion Motor Handoff

AUTHOR v0.10.26 preserves the v0.10.25 qualified training stack and fixes live runtime handling for explicit connected-region exit tasks such as `Leave the kitchen.`

The semantic task contract remains coordinate/object-ID free. When an ExitRegion task is active in the Discovery world, the private motor layer may select a physical Door, approach/open/cross it using learned skills, and the simulator region verifier decides completion. Raw LLM `MoveAhead` wandering is overridden by the verified motor handoff.

See `AUTHOR_v0.10.26_OPERATOR_README.md` for upgrade and runtime validation commands.
