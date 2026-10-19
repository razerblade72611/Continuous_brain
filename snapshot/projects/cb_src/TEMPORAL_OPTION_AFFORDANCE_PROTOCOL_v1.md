# Continuous Brain v0.21.1 — Temporal Option Affordance Diagnostic

v0.21.0 cleanly separated the system:
- memory->goal decoding: 100% TRAIN and DEV;
- donor memory->donor goal: 100%;
- zero memory->exact selector baseline: 100%;
- immediate goal-matching option coverage: 48.4375%;
- autonomous remembered-goal commitment: 62.5%;
- selector-only commitment: 48.4375%.

So memory is not the bottleneck. The remaining question is whether the six
existing temporal options contain the desired LEFT/RIGHT behavior later in
their execution even when their first primitive action does not show it.

This READ-ONLY diagnostic reuses the already-spent v0.21 DEV conditions
1600..1727. It trains nothing and opens no new DEV or TEST window.

At each identical paired delayed-decision state it executes all six frozen
options in the real environment and records immediate action, first lateral
action, temporal commitment, completion, duration, collisions and reward.

Decision fork:
- If >=90% of DEV members AND >=90% of paired conditions have an existing
  option that eventually makes the correct lateral commitment, the next build
  is a memory/goal-conditioned OPTION SELECTOR over the frozen repertoire.
- Otherwise, another selector cannot solve the missing behavior; the next build
  expands/relearns goal-diverse temporal options while preserving admitted
  memory.

This is not another Detour planner repair and does not reopen v0.19.
