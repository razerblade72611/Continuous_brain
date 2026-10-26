# A2-A0 closeout — early-retention triage (recorded 2026-10-25, before A2 Phase 1 run 2)

Immutable record agreed by Claude and Sol. Nothing below changes after Phase 1 results are seen; later findings go in a
new document that cites this one.

## Protocol
- **Purpose:** engineering triage before spending on the full A2 sweep — detect catastrophic early forgetting. It cannot
  establish the cause of forgetting and is not a scientific acceptance test.
- **Configurations:** the ten A2 Phase-1 configs — L lr {1e-4, 3e-4} × KL {0.3, 1.0}, night 4; T layer {7, 14, 21}, lr 1e-2,
  night 4; S layer {7, 14, 21}, lr 100, top_t 64, fitted keys, night 4.
- **Run:** dev seeds 0 and 1, 5 lives, checkpoints after lives 1, 2, 3, 5.
- **Measures:** acquisition = life-1 template recall at checkpoint 1 (not the A2 selector's (a1+a10)/2; no a10 = a1
  fallback); retention = life-1 forced choice at checkpoint 5 (50 candidates, nominal chance 0.2/10); competence = both
  monitors at checkpoint 5, mean perplexity ratio ≤ 1.02 and no seed > 1.05.
- **Decision rule (frozen before any A0 data):**
  - **GO** — some config acquires ≥ 6/10 mean, retains ≥ 3/10 mean and ≥ 2/10 in each seed, and is in the band →
    continue A2 Phase 1 to 25 lives.
  - **STOP** — at least one config acquires inside the band, every acquiring in-band config is ≤ 1/10 in each seed, and
    no other in-band config reaches ≥ 2/10 mean → stop the hyperparameter-sweep direction.
  - **INCONCLUSIVE** — anything else, including missing or invalid evidence → complete Phase 1 to 25 lives with the
    original selector; no automatic Phases 2 or 3.
- **Clarification made before any data:** STOP requires at least one acquiring config inside the band. An acquirer that
  retains but breaks competence gives INCONCLUSIVE, never a vacuous STOP.

## Execution
- **Code:** being/v3-runner-fix @ 28321c3 (experiment source identical to 95cad33; runner repairs R1–R5).
- **Backbone:** Qwen3-0.6B, SHA-256 `c75424fb5370f0d514a05d7c640e6c2e1e4b504564e4d48cb2ddedb71609e0a0` — matched the locked
  hash at preflight.
- **Hardware:** RunPod NVIDIA A40 48 GB, CA-MTL-1, `--parallel 5`. Benchmark at 1/3/6 recommended 6; we chose 5 because
  real per-worker peaks of 5.4–6.8 GB would have left only ~5 GB free at 6, below the 6 GB reserve. Observed 34.6 / 46.1 GB
  in use with 5 running.
- **Storage:** network volume 29qhoqghqf mounted at `/workspace` (nfs4,
  `ca-mtl-1-vast.runpod.net:/runpodfs/networkvolumes/29qhoqghqf`).
- **Artifact:** `results_stageA2_A0.tar.gz`, SHA-256 `cfe331f6b1962ef652290510cf54ecf0ff794eeeed61d1a2c271f48acebf7945`,
  22 files, re-read verified on the network volume.

## Result (unmodified)
**Verdict: INCONCLUSIVE.** No configuration satisfied GO.

| config | acq (mean /10) | ret @5 (mean, min seed) | band | ppl ratio eval / modern |
|---|---|---|---|---|
| L kl0.3 lr1e-4 | 8.5 | 0.5 (0) | ✓ | 0.613 / 0.639 |
| L kl1.0 lr1e-4 | 8.0 | 1.0 (0) | ✓ | 0.671 / 0.688 |
| L kl0.3 lr3e-4 | 9.0 | 0.5 (0) | ✓ | 0.661 / 0.674 |
| L kl1.0 lr3e-4 | 9.5 | 0.5 (0) | ✓ | 0.710 / 0.720 |
| T layer7 | 2.0 | 1.0 (0) | ✓ | 0.685 / 0.721 |
| T layer14 | 5.5 | 1.0 (0) | ✓ | 0.670 / 0.724 |
| T layer21 | 3.5 | 0.5 (0) | ✓ | 0.741 / 0.784 |
| S layer7 | 1.5 | 1.5 (1) | ✗ | 2.089 / 2.495 |
| S layer14 | 2.5 | 0.5 (0) | ✗ | 1.296 / 1.465 |
| S layer21 | 0.0 | 3.0 (3) | ✗ | 1.006 / 1.082 |

**Why INCONCLUSIVE rather than STOP:** all four L configs are competent acquirers. L kl1.0 lr1e-4 scored 0 and 2 across
the two seeds at life 5, so the STOP condition (every competent acquirer ≤ 1/10 in each seed) failed by one
forced-choice item. No other in-band config reached ≥ 2/10 mean.

## Interpretation recorded before A1
No configuration satisfied GO. Four LoRA configurations acquired well but showed severe early loss of forced-choice
accessibility. S layer 21 showed elevated forced-choice accuracy without demonstrated initial generation, and failed
competence: its modern ratio of 1.082 fails both the 1.02 mean limit and the 1.05 per-seed limit. The locked rule gave
INCONCLUSIVE, so only the original 25-life Phase 1 is authorized as the next development step. No scientific threshold
is changed.

- **The single 2/10 seed** is not persuasive evidence of sustained retention, but is not dismissed until the wiped
  controls have been examined.
- **S layer 21's 3/10 at life 5** is described as above-nominal-chance forced-choice performance of uncertain origin.
  It is not "retained discriminative signal": the backbone's answer priors are not uniform over the 50 candidates.

## Known measurement limitation (recorded, not applied retroactively)
Competence ratios of 0.61–0.78 for L and T show that the monitor is dominated by adaptation to the background-rehearsal
register. There is no literal overlap between the competence sets and the background text, and the first Stage A showed
the same effect for T and G. The ≤ 1.02 band is therefore insensitive to mild damage. Hypothesis, not established
cause.

Proposed for future gates only: a rehearsal-only control (same background stream and optimization budget, no new facts)
and held-out task competence alongside perplexity. Not retrofitted into A0 or Phase 1 run 2.

## Planned descriptive analyses (declared before inspection)
`scripts/a0_describe.py` reports the following for all ten configs and both seeds. The output is saved as
`results/dev3/stageA2_A0_describe.json`.
- wiped (no-memory) forced choice and mean strongest-rival margin on the life-1 facts — the answer prior;
- trained forced choice and margin at checkpoints 1, 2, 3, 5;
- trained-minus-wiped margin change at each checkpoint;
- which facts are right at life 5, and whether the wiped model already got them right.

These are descriptive only and cannot change the verdict.

## Lineage
- **Original run unavailable:** the original 25-life Stage A2 artifacts are on network volume `btxeuf9tm0`
  (CA-MTL-3) and are unavailable (no capacity; no S3 API in that DC). It is still unverified whether that pod used the
  volume.
- **Phase 1 run 2 is separate:** it writes to `results/dev3/stageA2_A1run2`. If the original files are recovered, both
  lineages are preserved, never merged. Recovered files stay quarantined until provenance is verified.
- **How to describe the Phase 1 outcome:** a configuration selected by Phase 1 is a development-screen winner, not a
  scientifically successful memory system.

## Next-direction note (Sol, agreed by Claude)
If the 25-life Phase 1 again shows acquisition followed by near-total inaccessibility, the next research effort shifts
to controlled associative-binding and interference experiments rather than extending the hyperparameter sweep.
