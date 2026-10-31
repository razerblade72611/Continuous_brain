# CONS-0A rerun (Amendment 2): operations log

## Attempt 1: 2026-10-31, H100 SXM (CA-MTL-1), code 271b39e, launcher e12ebd7. Stopped by the fail-closed prep check.

The A40 was out of stock in CA-MTL-1 for about 1.5 h, so the attempt ran on the only other GPU offered there.

- **Preflight:** model manifest OK. Run-1 results on the volume were moved aside and not resumed.
- **Calibration (QA family, seed 6):**

  | E | Template | Paraphrase | ppl (eval / modern) | Near-miss confabulation | Passed |
  |---|---|---|---|---|---|
  | 4 | 0.70 | 0.75 | — | — | fail |
  | 8 | 1.00 | 0.875 | 0.79 / 0.85 | 1.0 | **pass** |
  | 16 | 1.00 | 0.94 | — | — | pass |

  - E = 8 was chosen.
  - N's near-miss confabulation was 0.0, so no contract revision was needed.
  - The projection was 12,442 s.
- **prep, seed 2:** **42/52 reconstructions exact**, so prep FAILED CLOSED and nothing trained.
  - On the A40, run 1 reconstructed 52/52 for every seed.
  - **Finding (Sol's wording):** a hardware- or backend-dependent reconstruction failure under the tested
    configurations. bf16 Hopper-vs-Ampere arithmetic is not established as the sole cause; kernels or library versions
    may contribute. The neural episodic store has no demonstrated portability across execution conditions.
  - Portability will be a separate investigation. The stored adapters are not modified during CONS-0A.
  - This bears on the being's design: a store must survive hardware changes.
- **Bundle:** sha256 7b2ce083…3f74, held on David's PC. The pod was removed.

## Launcher change (no change to code 271b39e)
- **Results key:** results are tied to the pair (commit, replay mode).
- **GPU record:** the GPU name is written to results/cons0a/GPU.txt.
- **`CB_REPLAY=run1`** (opt-in; default `prep`):
  - Uses run 1's committed A40 prep outputs (results/cons0a_run1/seed*/replay.json; sha256 pinned in the launcher;
    52/52 exact, errors []) instead of prep.
  - stage_prep is unchanged since run 1, and with 52/52 exact an A40 re-prep is determined byte for byte.
  - **Provenance check:** each replay's recorded bank sha256 must equal the hash of the committed CLS-0B bank, and its
    model manifest must equal the CLS-0B manifest.
  - **Sol: GO for option B** (2026-10-31), under these conditions:
    - only the reconstruction inputs are reused; QA, calibration, optimiser, training, fresh evaluation and analysis
      all run under Amendment 2;
    - A1–A5 and the merge gates are unchanged;
    - this is **not** an independent confirmation of run 1;
    - the claim becomes "consolidation from previously verified neural episodic reconstructions", not end-to-end
      decoding plus learning on one GPU;
    - an I-QA advantage over S-QA alone does not authorise CONS-0B.
