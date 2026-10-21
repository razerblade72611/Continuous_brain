# Gate v2 sealed run: post-hoc notes

These notes were written after the sealed result. The locked `PREREG_v2.md` (r5, sha256 `f6e1a0c4…8713`) is not edited.

1. **Pool size correction.** PREREG says "values drawn from an 80-word pool" and "chance is 1/80". The code's `pool_ids` keeps only single-token words, which gives **69** words. The dev and sealed results both record `pool_size` 69, so chance on forced choice is **1/69**. The test, the gate and the result don't change; only the description was wrong.
2. **Run record.** The family was opened at `bcc82f6` at 05:57:06 UTC on 2026-10-21. It ran in two planned segments: seeds 1000–1012, then seeds 1013–1019 via `--resume`. There were no unplanned interruptions. Full result: `results/v2_sealed.json`, sha256 `3dbe10939b489119d50095e2d6433c6df27de1e773568598c462f974524d576c`. Outcome: **all 13 checks pass**.
