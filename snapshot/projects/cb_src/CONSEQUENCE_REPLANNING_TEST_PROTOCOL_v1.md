# Continuous Brain v0.25.2 — Frozen Consequence-Replanning TEST

## Authorization

v0.25.0 remains formally recorded as DEV FAIL because its original planner
ranking accidentally bypassed the intended surprise manipulation.

v0.25.1 repaired only that experimental construction error:
- no retraining;
- same trained consequence-model checkpoint;
- same spent DEV window;
- utility-first plan ranking;
- explicit surprise-exposure invariant.

v0.25.1 passed every repaired DEV criterion:

- memory-selected goal: 1.000
- overall replanning success: 1.000
- OPEN replanning success: 1.000
- KNOWN_BLOCK replanning success: 1.000
- SURPRISE_BLOCK replanning success: 1.000
- SURPRISE initial target-direct exposure: 1.000
- SURPRISE plan-once success: 0.000
- SURPRISE mismatch detection: 1.000
- SURPRISE replanning trigger: 1.000
- FIRST/SECOND goal+plan flip: 1.000
- KNOWN_BLOCK consequence-aware override: 1.000
- SURPRISE replanning advantage: +1.000

Route TEST remained untouched.

## Purpose

Perform the single frozen admission run on route TEST:

- condition IDs 2304..2559
- 256 conditions
- both FIRST and SECOND episodic queries
- OPEN / KNOWN_BLOCK / SURPRISE_BLOCK all represented

No parameter is trained or modified.

The exact v0.25.0 consequence-model checkpoint and v0.25.1 utility-first planner
are used.

After this run, route TEST is spent regardless of outcome.

## Frozen admission gate

Consequence-model fidelity:
- next-node accuracy >= .995
- next-blocked accuracy >= .99
- cost MAE <= .04
- success accuracy >= .995
- validity accuracy >= .995

Memory:
- selected episodic goal >= .99

Behavior:
- overall replanning success >= .95
- OPEN success >= .95
- KNOWN_BLOCK success >= .95
- SURPRISE_BLOCK success >= .95

Surprise causality:
- initial target-direct exposure >= .95
- plan-once success <= .10
- mismatch detection >= .95
- replanning trigger >= .95
- replanning advantage >= .80

Goal/consequence causality:
- FIRST vs SECOND changes goal and initial plan >= .95
- known blockage overrides naive target-side approach >= .95

## Preservation

Read-only verification covers:
- historical InfantBrain;
- admitted memory checkpoint;
- remembered-goal decoder;
- v0.23 episodic reader;
- v0.25 consequence-model checkpoint;
- v0.25.1 DEV repair result.

Historical PlanningCortex/world model remain untouched.

## Meaning of PASS

A PASS supports the following narrow functional claim on the sealed new route
family:

selected episodic content
 -> explicit remembered goal
 -> learned short-horizon consequence model
 -> multi-macro plan
 -> cached predicted future
 -> real action
 -> predicted/actual mismatch
 -> stale-plan invalidation
 -> replanning
 -> successful recovery

It does not establish 3-D planning, broad general intelligence, or
consciousness.

## Stop rule

After this run:
- route TEST 2304..2559 is spent;
- do not tune v0.25 against it.

If PASS:
- freeze and archive the consequence/replanning contract;
- stop symbolic planning assays for this phase;
- move to richer embodiment, preferably AI2-THOR / Discovery World.

If FAIL:
- retain the formal TEST FAIL;
- do not tune against TEST;
- take the architectural lesson into a new embodied development family.
