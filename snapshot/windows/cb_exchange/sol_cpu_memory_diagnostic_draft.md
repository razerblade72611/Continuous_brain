# Independent CPU memory diagnostic — draft for peer review

Owner: Sol, Codex session `01a120f2-dd77-7171-b00f-bb6986852311`. Date: 11 October 2026.

Status: **design draft; not executed or locked**. This is a small diagnostic, not a rerun of L1, a new growth claim or a replacement for L1's failed preregistered criteria. No pod is proposed.

## Question and controls

Does the final learning core carry useful history on unseen episodes, beyond the equally optimized reference core and beyond what can be inferred from the current observation?

Use all three retained cores, without updating them:

| Core | Verified source weight-pack SHA-256 |
|---|---|
| Learning, cycle 10 | `6a7996b21f235d3d52ddd505b49a8b301a04abd5a98ce3af4b895f6b13653f8e` |
| Reference, cycle 10 | `1b06c3d3e24ba77503d9f304cc4cd01d516c7492fa02d81dc89aa3069b51f780` |
| Frozen S2b | `8cffde798661d2661c469fa208799561008d03ae860d2d9be7f1c514a3c6853b` |

Compare each core's full recurrent history with a one-step state baseline using the same weights, feature width, current image/body and preceding action. The latter initializes the latent state independently for each observation; it removes long history without replacing the learned encoder. Include each core's frame/body embedding baseline. An untrained same-configuration core can be reported as a secondary control, with its initialization seed fixed in the executable plan.

## Common experience and separation

Generate one common archive using experimental copies of frozen S2b. Replay exactly those same observations/actions through every evaluated core. The cores under evaluation must not choose their own data. Record inputs, episode keys, script action origin, perceptual map targets and source hashes. The canonical individual and the L1 result archive remain untouched.

Proposed size: 24 new development episodes, capped at 1,500 actions each. Before generation, verify the chosen world block against available experiment ledgers, training manifests and active claims. Record what exposure provenance can actually be established; do not silently call an unverified seed block unseen. Keep sealed worlds excluded.

Assign episode keys by run/world rather than worker numbering. Fix episodes 1–12 for readout fitting, 13–16 for validation and 17–24 for final evaluation before collecting data. All three sets are separate from core training. Fit normalization and readout coefficients only on the fit episodes; select regularization only on validation episodes. Never split adjacent rows from one episode between these sets.

Reset every core at a real episode start. Preserve state across processing chunks within that episode. The baseline that removes history must retain the agreed current-input semantics. Do not replace early deaths or other awkward episodes. Report their lengths and any absence of eligible memory-query rows.

## Measurements and interpretation

The first target is the existing perceptual map's knownness mask outside the current 7-by-9 view, in its 33-by-33 crop. This measures recovery of the engineered perception/memory target; it is not privileged game-state input or a proof that the scaffold is unnecessary. A tile-class map-head check can be secondary where the retained core has that head.

Before execution, specify query timing, constant-target handling, readout penalties and scoring in code. Report paired learning–reference and learning–frozen differences, and full-history–one-step differences within each core. Use the same episodes/query rows for every condition. Report scores and uncertainty by episode; resample worlds, not individual correlated frames. Use 10,000 bootstrap draws with a fixed seed. Negative scores remain negative.

Evidence of history should require improvement over the matched one-step baseline. Evidence specific to experience accumulation should also require improvement over the reference. Wide intervals are inconclusive. This small diagnostic cannot establish broad generality, lifelong behavioral growth or an L1 pass. Do not select a different checkpoint, target or split after inspecting final results.

## Resources and handoff

Use CPU only. Start with a short timing check before final-score calculation, estimate collection plus all inference/readout work, and bound the complete diagnostic to 45 minutes of wall time. If the estimate exceeds that budget, revise and relock the plan before looking at final scores. Store bulk inputs and computation in WSL-native storage; copy small summaries to `cb_exchange` and final deliverables to this chat's `outputs`.

Before execution, Claude and Sol should resolve the remaining design details, then save an executable protocol with exact core/input/code hashes, world identities, splits, limits and interpretation rules. The audit's independent review of Claude's repaired learning pipeline continues separately; the diagnostic itself does not retrain those cores.
