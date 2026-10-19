# Continuous Brain v0.37.1 — Completion attribution diagnostic

## Preserved starting result

v0.37.0 is a formal DEV FAIL. The user's completed run is:

`state/compositional_autobiographical_goals_v1/runs/20260920T055442Z_8b224fc43c`

| Reported measure | Count / 128 | Rate |
|---|---:|---:|
| Both normal clauses complete | 90 | 0.703125 |
| Both simulated capability gains at least 0.35 | 91 | 0.7109375 |
| Drop-second control meets both original target predicates | 11 | 0.0859375 |
| Misbind-second control meets both original target predicates | 9 | 0.0703125 |
| Normal completion advantage over drop-second | 79 | 0.6171875 |
| Normal completion advantage over misbind-second | 81 | 0.6328125 |

Both controls preserved the missing target's capability exactly. The four
failed gates are the two zero-control-success gates and the two >=0.70
advantage gates. The original thresholds are unchanged. TEST 140000–140127
and its context trace ranges remain sealed. v0.36 TEST is not rerun.

## Why this diagnostic is needed

The v0.37.0 goal author reads bootstrap autobiography before three unrelated
stable/direct interruptions. The executor then checks completion using the
post-interruption autobiographical estimates. It does not require a new episode
targeting the named primitive before accepting that estimate as completion.

Each context starts with 12 records. Appending the stable interruption makes 13
stored records; the normal reader uses only the latest 12. Thus the event both
adds stable-primitive evidence and removes the oldest record from the readout.
The existing bootstrap ordering makes that oldest record a fragile-primitive
record. A joint recurrent readout can change either reliability estimate when
the history changes. Structural neutrality does not imply independence of its
two outputs under nonzero history.

After the interruption, neither control executes any action in the second
target's context. Consequently, with frozen deterministic models, its final
control readout should equal its post-interruption readout. The successful
controls therefore point to a target that was already satisfied at this
boundary. This is a source-based inference until replay is checked against the
saved subject records. No subject-specific replay results are claimed here.

The original software tests did not exercise unrelated-history changes to the
completion estimate. This package adds that regression fixture and measures
the actual trained model's behavior in the user's environment.

## Record audit

Read result.json, subjects.jsonl, integrity_before.json and integrity_after.json
from the exact preserved run. Require the recorded FAIL, original thresholds,
all 128 DEV IDs, and unchanged original sources/checkpoints. Recompute the old
summary and require exact equality with the saved report.

Independently reconstruct goal predicates and capability gains from the saved
numerical fields. Check all normal and control action bindings, original
targets, episode counts and the equality of the two untouched-context readouts.
Reject contradictory records rather than silently classifying them.

Report separately:

- target completion;
- capability-gain qualification;
- their per-subject conjunction;
- completion with no practice of one clause;
- completion with a subthreshold capability gain;
- exhaustion at clause 1 or clause 2;
- each control's success and missing-context readout.

The 90/128 and 91/128 source rates do not establish the size of their overlap.
The diagnostic reports that overlap without creating a replacement admission
criterion.

## Frozen interruption replay

For each of the same 128 already-used DEV subjects:

1. Rebuild the exact original three bootstrap histories with the original model.
2. Require the original reliability estimates and authored goal to reproduce.
3. Re-execute only the original stable/direct interruption in each context.
4. Record each original clause's estimate and capability before and after it.
5. Compare the second context's post-interruption pair with BOTH control pairs
   in the saved run.
6. Reconstruct each control's joint predicate using its saved first-clause
   result and the replayed second-clause result.

No target-development episode, 32-episode control, full DEV trial, or TEST trial
is rerun. Neural weights remain frozen. Replay acts only on in-memory copies.
The historical source run remains read-only.

An observed control success is explained by the interruption only if the replay
is exact, the original target crosses at that interruption, no target episode
occurred there, and the target capability remained unchanged. Any reproduction
mismatch keeps the overall attribution unresolved. The program does not hide
small mismatches with a numerical tolerance.

## Descriptive history interventions

For both selected contexts, read the same frozen model under four histories:

| Probe | Valid records | What is supplied |
|---|---:|---|
| bootstrap_12 | 12 | Original bootstrap |
| drop_oldest_only_11 | 11 | Original bootstrap excluding its oldest record |
| retain_all_plus_interruption_13 | 13 | Entire bootstrap plus the interruption |
| actual_rolling_window_12 | 12 | Latest 12 after the interruption |

For each probe, report both estimates, original target crossings, valid record
count and target-primitive record count. The 11-record probe uses the usual
12-slot padded input; the 13-record probe explicitly uses a 13-slot input.
They alter effective sequence length and are descriptive interventions, not
validated changes to the production memory rule. A recurrent model can have
nonlinear interactions among eviction, added evidence and sequence length.
These probes do not claim a unique additive decomposition of those effects.

## Output and interpretation

This is a diagnostic, never a new scientific PASS. It writes:

- source_record_audit.json: independent per-subject record audit;
- boundary_replay.jsonl: before/after estimates and memory probes;
- result.json: counts, reproduction checks and attribution status;
- integrity_before.json / integrity_after.json;
- run.log and result_bundle.zip.

Every invocation receives a new directory under
state/compositional_goal_completion_diagnostic_v1/runs/. Source artifacts and
protected files are hashed before and after. Successful execution prints
DIAGNOSTIC_COMPLETE; that label says only that the diagnostic completed.
development_gate_passed remains null, admission_performed remains false, and
historical_v0370_status remains DEV_FAIL.

If all observed control successes are reproduced as interruption crossings,
the next design review should address completion attribution and history-window
effects. It must also inspect the per-subject target/capability conjunction and
normal-arm timeouts. This diagnostic does not establish that repairing the
control false positives will make the normal arm pass.

If reproduction fails or some control successes remain unexplained, preserve
the report and investigate that discrepancy first. Do not lower thresholds,
increase the horizon, replace the self-estimate with evaluator-side capability,
or open TEST on the strength of this diagnostic.
