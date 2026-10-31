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

## Volume-free launcher (scripts/cb_cons0a_portable.sh), 2026-10-31
- **Why.** CA-MTL-1 had no A40, no H100 and no CPU pods for hours, so the volume's venv versions could not even be
  read (scripts/cb_env_probe.sh failed with no instances).
- **Sol: no objection,** on condition of these three safeguards, all implemented:
  1. **Checkpoints kept.** All 24 cortex checkpoints are bundled with a sha256 list and verified on the PC before the
     pod is removed. If the job finished but the download did not verify, the pod is stopped, not removed.
  2. **Environment as run identity.** Every launch is a fresh pod and a fresh campaign, never resumed or mixed across
     environments. The platform is recorded before calibration (GPU.txt, ENV.txt = full pip freeze, cb_boot.log). The
     platform must not be changed in response to results.
  3. **GPU feasibility check before calibration:**
     - CUDA is selected;
     - Qwen3-8B loads in bf16;
     - all 252 cortex sites are installed, in fp32;
     - one step gives a finite loss and nonzero gradients.
     Nothing falls back to CPU, and nothing changes precision.
- **Software stack.**
  - Pinned: torch 2.14.1 (the build matched to the host driver) and transformers 5.21.0, the versions
    tests/test_cons0a.py passes with.
  - Before the job, the 12 CONS-0A tests run on the pod, on CPU.
  - Qwen/Qwen3-8B is fetched at revision b968826d, whose 12 manifest files are exactly the CLS-0B manifest's.
  - The Qwen3-0.6B tokenizer is fetched; the stream-design check guards it.
- **GPU order:** A40, A6000, 6000 Ada, L40S, by availability. This is a fixed policy, not chosen per result.
- **Not an independent confirmation.** Library versions differ from run 1 and CLS-0B (unknown). This changes numerics,
  not the within-run matched comparisons. No reconstruction happens on this stack (CB_REPLAY=run1 is forced).
- **Input audit (David asked whether the portable inputs are as complete as the volume's).**
  - **Byte-verified on both paths:**
    - the code (pinned 271b39e);
    - the CLS-0B banks/inputs (FILES.sha256);
    - run 1's replays (pinned sha256 plus provenance);
    - the 12 Qwen3-8B files (CLS-0B manifest);
    - data/ and results/dev3/bind_i1 (in the commit).
  - **Stream tokenizer.** The volume's copy was never hash-verified; it was only design-checked on dev seeds, and not
    on calibration seed 6. The portable launcher now requires the 4 downloaded tokenizer files to equal the CLS-0B
    manifest's (HF: identical git blobs to Qwen3-8B; tokenizer.json lfs sha256 aeb13307… = the manifest). On this
    input the portable path is the stronger one.
  - **Not identical:** the Python stack and the GPU (disclosed above).
