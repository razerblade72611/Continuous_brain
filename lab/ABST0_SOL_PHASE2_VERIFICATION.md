# Sol: independent phase-2 verification at 25966d4

**Scientific verdict confirmed: ABST0_NO_ADVANCE.** D1 and D4 pass; D2, D3 and the mechanism gate fail. The prior ABST0_INVALID from Sol's initial analyzer was a normalization-contract mismatch, not a demonstrated defect in the phase-2 run.

Inputs were the committed JSON results at `25966d4b0f65d1e64a85e15fc8f2240486e31322`, under `results/abst0/phase2`. The recorded bundle SHA-256 is `5a3c7db344f06ab868cd1f28db2f9f86171836ab2b55b453c43a3ef4e6b4cd10`; the compressed bundle itself was not available in this session to rehash. Analysis used the committed JSON snapshot. No live infrastructure or keys were accessed.

## Cause and correction

The pre-existing [CLS-0C-B answer contract](https://github.com/razerblade72611/Continuous_brain/blob/0cf967e0e2d96a80c99a6414efeadba323456c68/lab/CLS0C_B_PROTOCOL.md#answer-contract-and-scoring-evaluator-side) says lowercase, punctuation stripped, leading articles removed. ABST-0 referred to normalized equality without restating that inherited definition.

The original Sol normalizer retained the leading article. Across all 140 evaluation files and 29,505 row records, the resulting 1,472 disagreements were:

| Category | Row records changed from substantive to correct |
|---|---:|
| POS | 882 |
| FIRST | 82 |
| RET | 434 |
| TPOST | 56 |
| UPD | 18 |
| Total | 1,472 |

These are row records, including repeated category views and merged/unmerged twins; they are not 1,472 independent questions. They explain all 55 arm-by-sleep validation entries with mismatched stored outcomes. Removing a leading article eliminates every discrepancy. No negative row changes class.

`normalized()` now removes a single leading `a`, `an` or `the` followed by whitespace, case-insensitively, before its existing ASCII-alphanumeric normalization. The code and output identify the inherited contract. Tests include `a draembrernhound` versus `draembrernhound`, all three articles in both cases, article-prefixed UNKNOWN, UNKNOWN explanations, and non-article words such as `another` and `theme`.

This is a clarification of the existing scoring contract. It changes neither the frozen scorer, training data, thresholds, labels, nor CODE_HASH. Claude should make that inheritance explicit in the ABST-0 protocol text and record this compatibility correction without rewriting the historical as-written output.

## Robustness argument: true for this run, not a general invariance

Article removal can change a negative answer's class: `the UNKNOWN` changes from substantive to UNKNOWN. D3 also includes positive TPOST recall, which can change when a leading article is removed. Therefore the statement that an article cannot turn an answer into UNKNOWN is false in general.

In this actual result set, all negative classes are unchanged. D2 and familiar TPRE fail under either normalization; the G-minus-U HNM contrast is unchanged. TPOST scores change, but D3 already fails TPRE at sleeps 2 and 3. D1=true, D2=false, D3=false, D4=true under both row-grading rules. Thus the scientific NO_ADVANCE conclusion is robust. The analyzer's top-level label changes from INVALID to NO_ADVANCE because its earlier compatibility validation is now corrected.

Key pooled U values (equal-weight seed means):

| Sleep | HNM substantive | FRESH substantive | Familiar TPRE UNKNOWN |
|---:|---:|---:|---:|
| 1 | 0.7844827586 | 0.8625 | 0.1875 |
| 2 | 0.64375 | 0.6000 | 0.40625 |
| 3 | 0.82500 | 0.7375 | 0.25000 |
| 4 | 0.88125 | 0.8750 | — |
| 5 | 0.82500 | 0.6625 | — |

Sleep-1 TPRE includes only the already familiar transition entities; the unfamiliar cohort is separate. The D3 TPRE gate applies at sleeps 2 and 3. Their A/H UNKNOWN rates are 0.375/0.4375 at sleep 2 and 0.3125/0.1875 at sleep 3, all below 0.80.

Sleep-5 RET is 0.9000000000 for U versus 0.9166666667 for P. Every seed has zero exposed A-transition cells, so the predeclared inadequate-exposure note applies. The run does not adequately test reversal of explicitly trained UNKNOWN on those cells.

## Full comparisons and numerator/denominator checks

The corrected independent result has no errors and no invalid arms. All 60 trained-arm/sleep twin checks pass, plus 20 base-only validations.

| Comparison with the saved production report | Checks | Differences |
|---|---:|---:|
| Pooled scalar metrics | 460 | 0 |
| Outcome rates | 912 | 0 |
| Twin integrity fields | 420 | 0 |
| D1–D4 and functional result | 5 | 0 |
| Mechanism, invalid arms, verdict | 3 | 0 |
| Total | 1,800 | 0 |

Separate regrouping of the emitted row outcomes agrees with all **5,280 per-seed metric numerator/denominator pairs** and **3,496 per-seed outcome-count tables**. The 400 descriptive update fractions also agree with the emitted per-update old/new flags and outcomes. These checks are against the input row records; the saved production analysis report does not itself contain per-seed count/denominator tables.

The full corrected JSON also equals Claude's committed `sol_analysis_with_article_rule.json` after excluding only the new top-level `normalization_contract` citation. The historical as-written and scratch-corrected outputs were left untouched.

## One log discrepancy: three improving seeds, not two

The phase-2 protocol log at `25966d4` says U has lower HNM confabulation than G in 2/4 seeds. The row counts give **3/4**:

| Seed | U HNM substantive | G HNM substantive | G minus U |
|---:|---:|---:|---:|
| 12 | 30/40 | 32/40 | +0.050 |
| 13 | 36/40 | 35/40 | −0.025 |
| 14 | 31/40 | 32/40 | +0.025 |
| 15 | 35/40 | 40/40 | +0.125 |

The equal-weight mean difference is **+0.04375**, below the required +0.30. The three-seed direction condition passes; the pooled-size condition fails. The mechanism verdict remains FAIL. This is a descriptive log correction; no discrepancy was found in the compared production scalar metrics or gate verdicts.

Interpretation remains limited to this experiment: recall and retention survived, but the required evidence-sensitive abstention boundary did not emerge. Audit-label noise was measured; its causal contribution to failure is not bounded merely by its candidate error percentage.
