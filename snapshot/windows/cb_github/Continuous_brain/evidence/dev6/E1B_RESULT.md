# E1b — store-authority positive control (ENGINEERING_ONLY)

Report: `e1b_report.json` (format cb-e1b-store-authority/3), produced by `scripts/dev/e1b_store_authority.py --n 32 --e1-tag e1v2`. All runs used the `cloud-dev-claude-eng` slices (32 seeds per mechanism). The report reads only hashes, exit codes and attestation structure.

**Gate: PASS.** For all 13 (route, host) pairs in which a host received requests, replacing that host's response at the store level changed what the frozen evaluator did. The replacement is a REPLACE intervention through the runtime's intervention manager.

Rows (25):

| Outcome | Rows | Meaning |
|---|---:|---|
| CHANGED | 18 | result.json differs from the G0 route |
| ABORTED_AFTER_REPLACEMENT | 2 | the evaluator ran past the first replaced tick and then failed |
| UNCHANGED | 4 | 16 single-tick replacements of one API left the result byte-identical |
| NOT_TESTABLE | 1 | `goal_is_advanced` returned the same response for its first 64 calls |
| INFRA | 0 | |

The four UNCHANGED rows are robustness observations, not bypasses. Each host involved shows authority through another API or through tier 2:

- `update_explicit_intention` in the v0.34.1, v0.35.3 and v0.36.1 routes;
- `rollout` in the v0.36.1 route. Its first 64 calls build the first subject's bootstrap memory, and the downstream decisions were robust to replacing 16 of them.

**Weakest case, stated plainly:** in the v0.36.1 route, the v0.31 (`self_state`) host's store authority is shown only at tier 2. Replacing its whole response stream after the first response made the evaluator abort on a wrong-typed value. That shows the evaluator consumes what the store holds, but it does not show a graded dependence.
