# G0 Isolated Legacy Requalification — PROPOSED preregistration v1

Status: **PROPOSED — NOT APPROVED**. This document authorizes no scientific run by itself.

## Question
For each historically admitted mechanism whose source, evaluator, learned state and primary result are hash-recovered, does the frozen mechanism still satisfy the *same evaluator checks, thresholds and causal interventions* when evaluated in isolation on a fresh registered DEV family?

G0 is a requalification prerequisite to composition. It is not a new admission TEST and makes no consciousness claim.

## Frozen elements
- Historical module, evaluator and checkpoint/result files are identified by `legacy/g0_plan.json` and must hash-match before and after each run.
- Historical evaluator source is byte-identical. No threshold or intervention constant may be changed.
- Historical sealed TEST ranges remain spent and are never opened.
- The only runtime substitution permitted in first-wave recipes is the named module-level seed-family symbol (`TEST_IDS` or `TEST_ROUTE_IDS`).
- `--state-dir` is redirected to the new G0 run directory outside the legacy root.

## Fresh families (proposal)
`legacy/g0_seed_plan.json` defines one disjoint DEV family per mechanism. They are proposals, not registered/opened ranges until David approves this preregistration.

## Decision rule
For a mechanism with an approved recipe:
1. every pinned G0 artifact hash matches before execution;
2. the requested seed family is registered DEV, unspent, disjoint from historical/sealed ranges, and has exactly the historical evaluator population size;
3. the evaluator is imported from the pinned file and only the recipe's named seed symbol is replaced with that fresh range;
4. evaluator execution exits 0 and writes its raw result;
5. the raw result's original pass field is `true`, meaning the original threshold/check set was satisfied on the fresh family;
6. every pinned historical artifact still hash-matches and the legacy tree metadata fingerprint is unchanged after execution.

If all six hold, outer G0 status is `PASS`; otherwise `FAIL` or `BLOCKED` depending on whether execution occurred. The raw evaluator's legacy field names such as `test_admission_passed` are preserved only as raw evidence and are never relabeled as a newly opened TEST.

## First-wave exclusions
v0.24, v0.28, v0.26, v0.21/v0.22, working-memory v0.13-v0.19, and v0.29 stay blocked/control for the reasons in `legacy/g0_recipes.json`.

## Approval
David must explicitly approve `G0-LEGACY-REQUAL-v1` before a follow-up artifact changes its registry status from PROPOSED and registers any fresh DEV ranges.
