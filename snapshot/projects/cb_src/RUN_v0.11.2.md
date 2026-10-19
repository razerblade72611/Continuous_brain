# Run v0.11.2 — DirectML recurrent-cell smoke

Use the existing DirectML environment and the exact v0.11.0 expanded checkpoint.
No capacity expansion or curriculum recollection is required.

1. Install v0.11.2 with `pip install -e . --no-deps`.
2. Copy the exact expansion checkpoint to a disposable smoke branch.
3. Run the same 10-update `train-whole-brain` command with `--device dml`.
4. PASS requires completion of all 10 route steps, all route families exercised,
   and `legacy_tensors_changed == 0`.
5. Do not begin the 6000-update Phase A until this smoke passes.
