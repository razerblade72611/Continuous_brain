# Frozen-core memory diagnostic: passed, with a narrow claim

The final L1 learning core carries useful information from earlier observations on this test. Its separate diagnostic readout predicted whether a tile outside the current view had previously been seen, with 69.04% balanced accuracy. All five required comparisons passed the locked rule on all 16 eligible test worlds.

| Learning core with full history compared with | Gain in percentage points | One-sided 95% lower bound | One-sided 95% upper bound |
| --- | ---: | ---: | ---: |
| Same core, one observation and preceding action | +5.85 | +4.29 | +7.44 |
| Same core's image/body/preceding-action embedding | +2.82 | +1.37 | +4.22 |
| Final reference core with full history | +1.76 | +0.75 | +2.77 |
| Original frozen S2b core with full history | +2.54 | +1.26 | +3.81 |
| Constant spatial prior fitted on the fit worlds | +10.77 | +8.35 | +13.07 |

The underlying mean balanced accuracies were 69.04%, 63.19%, 66.22%, 67.28%, 66.50% and 58.27%, respectively. Balanced accuracy gives known and unknown tiles equal weight; each test world also receives equal weight. The lower and upper bounds are the fifth and ninety-fifth percentiles of 10,000 paired whole-world bootstrap means, rather than a two-sided 95% interval.

The plan and executable were locked before collecting any main life. Frozen S2b generated 48 common episodes: 24 fit, 8 validation and 16 test worlds. Readout penalties and normalization were selected without test labels. None of the three cores received updates. All early deaths were retained, no world was replaced, and three lives reached the predeclared 1,000-action cap. The diagnostic recorded 23,057 actions and completed once in 918.30 seconds, using local CPU only.

A separate verification recomputed every reported contrast and bound, checked 192 input/feature file hashes, confirmed finite features and readouts, and confirmed the source checkout was clean at b6a76ee. Core parameter hashes stayed unchanged. The exact plan, script, source-pack and output hashes are retained in the linked evidence.

This measurement uses the correct x/y map geometry. The old out-of-view mask admitted 14 currently visible tiles and excluded 14 hidden tiles; this separate defect was reported to Claude for correction. It does not change L1's behavioral result.

The secondary age breakdown is descriptive: full-history versus one-step balanced accuracy was 79.23% versus 68.83% for tiles last seen 1–10 actions ago; 69.26% versus 62.66% at 11–50; and 64.88% versus 61.17% at 51 or more. These strata reuse unknown-tile negatives and are correlated. They do not create additional independent passes.

The existing trained map heads remain weak at identifying the types of known hidden tiles: mean class-balanced accuracy was 17.55% for learning and 17.15% for reference, compared with 3.30% and 5.52% for their one-step controls. The primary result shows usable information in state, not a reliable replacement for the engineered map.

The claim is limited to this perceptual-map target and frozen S2b's experience distribution. It does not show general memory, planning, improved survival, technical continuity beyond the existing checks, or selfhood. Earlier seed exposure is not exhaustively documented, although the new episode files were never supplied to any core training. There are only 16 test worlds, and the uncertainty conditions on the fitted readouts rather than including all fitting-data variability. Feature widths differ between the frame embedding (4,625) and recurrent state (1,536); the same-core one-step control has the identical 1,536 width.

L1's full behavioral growth claim remains unestablished. Canonical S2b is unchanged, and learned-authored primitive action share remains 0%. This positive memory result supports continuing the agreed repaired L1b review and local rehearsal. It does not authorize an automatic pod launch or a canonical upgrade. The revised code, complete protocol, isolation checks and bounded compute budget still need to be finalized.

Result: [CPU_Memory_Diagnostic_Result.json](C:/Users/davei/Documents/Codex/2026-10-09/referenced-chatgpt-conversation-this-is-an/outputs/CPU_Memory_Diagnostic_Result.json). Independent verification: [CPU_Memory_Diagnostic_Verification.json](C:/Users/davei/Documents/Codex/2026-10-09/referenced-chatgpt-conversation-this-is-an/outputs/CPU_Memory_Diagnostic_Verification.json). Locked protocol: [CPU_Memory_Diagnostic_Protocol.md](C:/Users/davei/Documents/Codex/2026-10-09/referenced-chatgpt-conversation-this-is-an/outputs/CPU_Memory_Diagnostic_Protocol.md). Locked plan: [CPU_Memory_Diagnostic_Plan.json](C:/Users/davei/Documents/Codex/2026-10-09/referenced-chatgpt-conversation-this-is-an/outputs/CPU_Memory_Diagnostic_Plan.json).

Result SHA-256: 01e932f215c3219035002e107747fa254e517675e2cc2248b853024c546f7ef3.
Plan SHA-256: fc0d0d3f19d5764199455849f0b6f62af0884edf3b8d0d2b88a0f5d880976735.
Script SHA-256: a7fa1e7d8f9e81b89c4fb5587c44fe5ec17c98f45c46eefdd3d6b041d60f2e02.