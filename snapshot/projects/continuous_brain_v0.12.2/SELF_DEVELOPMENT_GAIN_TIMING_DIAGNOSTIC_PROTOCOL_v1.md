# v0.33.2 — Self-Estimate Gain Timing Diagnostic

v0.33.1 is preserved as a formal DEV FAIL.

Only one gate failed:

`new_autobiography_updates_self_estimate`

The failed metric asked whether the **final** autobiographical reliability was
still at least +0.12 above bootstrap in >=80% of subjects.

However, the controller graduates a fragile commitment at the moment its
autobiographical reliability crosses:

`bootstrap + 0.12`

After graduation, behavior changes from SCAFFOLD to DIRECT. Those later DIRECT
episodes themselves alter autobiography and can move the final reliability back
below the +0.12 line.

That creates a possible measurement mismatch:

- causal developmental event: **ever crossed +0.12**
- old gate: **finished above +0.12**

This read-only diagnostic reuses the already-spent DEV window 10000..10127 and
the exact frozen v0.31/v0.32 models. It performs no training and does not touch
TEST 20000..20127.

For every subject it records:
- baseline reliability;
- complete per-cycle reliability trace;
- peak gain;
- final gain;
- whether +0.12 was ever crossed;
- whether +0.12 remained true at the end;
- whether the subject graduated to DIRECT;
- whether it crossed then later fell below.

The key diagnostic prediction is:

`ever-crossed rate == graduation rate > final-above rate`

with:
- no graduation without a prior +0.12 crossing;
- no +0.12 crossing without graduation.

If true, the v0.33.1 lone failure is specifically a timing/readout mismatch:
new autobiography did update the self-estimate enough to cause the developmental
transition, but the final state no longer retained that same magnitude after
the behavior policy changed.

This diagnostic does not retroactively turn v0.33.1 into a PASS. Its result
determines whether a new DEV repair may replace the final-only gate with an
event-aligned causal gate while preserving the original failure.

TEST remains sealed.
