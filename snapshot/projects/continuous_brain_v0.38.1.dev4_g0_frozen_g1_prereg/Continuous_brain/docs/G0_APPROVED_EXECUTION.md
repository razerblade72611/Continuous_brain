# G0 approved execution protocol

`G0-LEGACY-REQUAL-v1` was explicitly approved by David on 2026-09-27.

Do not use `--run` until all seven read-only preflights report:
- `evidence_ok: true`
- `prereg_status: APPROVED`
- empty `blocked_reasons`

Scientific opening requires both:
- `--run`
- `--confirm-open G0-LEGACY-REQUAL-v1`

The runner marks the mechanism's fresh DEV family spent in `registry/seed_ranges.local.jsonl` before launching the historical evaluator. A crash or FAIL consumes that family. Do not delete or edit the ledger.

The runner also verifies a hash-pinned project-local Python dependency closure before opening a family and again after execution. Any mismatch blocks before opening or forces FAIL after execution.
