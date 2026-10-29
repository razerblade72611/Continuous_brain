# CLS-0C Phase A: read-only diagnosis of K1 addressing (frozen 2026-10-29, Claude + Sol)

## Purpose
Find out *why* K1 activates 58 of 80 near-miss cues in CLS-0B, and whether a single similarity threshold could fix it,
before building any remedy. The diagnostic is descriptive and read-only:
- no training;
- no writes to CLS-0B results;
- no change to CLS-0B thresholds or verdicts.

## Inputs
- **CLS-0B results:** the committed `results/cls0b_final/seed{2..5}.json`, verified against `FILES.sha256` on the pod.
  The diagnostic uses the episode texts (sha-checked) and the recorded decisions.
- **Embedder:** the same Qwen3-Embedding-0.6B. Its file manifest must equal the one CLS-0B recorded, otherwise the run
  stops.
- **Cues:** rebuilt with the frozen `scripts/cls0b.py` `build_cues`. Cue counts must equal the recorded `n_cues`.

## Validity gate (R)
- **Thresholds:** at every checkpoint, the recomputed K1 threshold must be within 2e-3 of the recorded one.
- **Decisions:** at least 99% of the recorded K1 decisions must reproduce. The comparison covers:
  - the actual routed episode ID per fact cue (None when declined), not merely whether it was correct (Sol);
  - activation per negative cue;
  - the ordered routed list per temporal cue.
- **If either fails:** the run is invalid (exit 3) and no conclusion is drawn.
- **Code check:** the identical pipeline reproduces the recorded K3 (TF-IDF) decisions exactly on all seeds and lives
  (`tests/test_cls0c_a.py`).

## Questions (life 5 primary; all lives recorded)
- **Q1:** for each near-miss and unseen cue, the top-5 candidates (eid, entity, attribute, time, similarity) and the
  relation of the top-1 candidate to the cue: same entity, same attribute, or other. Reported for all near-misses and
  for activated ones.
- **Q2:** top-1 similarity distributions per cue form, and the AUROC of correct template top-1 against near-miss top-1.
- **Q3 (decisive for calibration):** does ONE K1 threshold give template recall ≥ 0.95 with near-miss activation
  ≤ 0.10?
  - Template recall means the top-1 is correct and at or above the threshold.
  - Paraphrase ≥ 0.90 is also reported.
  - Per seed and pooled across seeds; the pooled result decides.
  - **Two levels are reported (Sol):**
    - *Narrow:* template ≥ 0.95 with near-miss ≤ 0.10.
    - *Full:* narrow, plus paraphrase ≥ 0.90 and unseen-person activation ≤ 0.10.
  - **If narrow is infeasible:** global-threshold tuning is eliminated as the primary CLS-0C remedy.
  - **If narrow is feasible but full is not:** this is reported, but not treated as a solution.
  - **If full is feasible:** threshold-only calibration stays a serious Phase B candidate. Development data is not a
    test, and this is an oracle feasibility calculation.
  - **Threshold sweep:** includes a finite reject-all value (2.0), and JSON output is strict.
- **Q4 (entity vs attribute):** for every stored fact (entity E, attribute A), compare the similarity of the episode to:
  - its own template cue;
  - E asked about every attribute never observed for E (attribute swap);
  - an unseen person asked about A (entity swap).

  The output reports whether the mean similarity drop from an entity swap exceeds the drop from an attribute swap.
  This is evidence of relative sensitivity that *supports* entity dominance; it does not prove it, because the two
  swaps change different tokens (Sol).
- **T (temporal):** top-2 candidates for the life-1, life-3 and latest probes.
  - Offline finding already in hand: in seeds 3 and 4, both misses retrieved both correct episodes.
  - So those misses are a readout/temporal-reasoning failure, not a retrieval failure.

## Cost and run
- **Hardware:** one A40 pod with the network volume.
- **Time:** about 5 min of embedding, with a 45-min cap.
- **Launcher:** `scripts/cb_cls0c_a_run.sh`, using a separate checkout pinned to the diagnostic commit. It
  downloads and verifies the bundle, then removes the pod.
- **Exit codes:** 0 valid, 3 invalid, 1 operational failure.
