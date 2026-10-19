# Continuous Brain v0.19.0 source review

Reviewed 2026-09-18. This is an audit record, not an installable model release.

**Decision: keep v0.19 closed at its declared 4800-update limit. Preserve its
protected dependencies and use the planning/option integration audit to choose
the next branch.** The uploaded code supports a narrower interpretation of the
reported success: strong response mapping on teacher-generated TRAIN sequences.
It does not establish autonomous temporal control.

## Evidence examined

- Uploaded `continuous_brain_v0.19.0_recurrent_option_progress_v1(1).zip`, SHA-256
  `9c9793287537653b3bc9f19c49d5a14be16143cfedd029be48f17b00163dbf86`.
- Uploaded `Pasted text(20260918-001347).txt`, containing the qualification,
  training, and final evaluations at updates 1600, 3200, and 4800.
- Previously recovered v0.12.2 source for the historical planning/option stack.

No trained v0.18/v0.19 weights or v0.19.1 audit package were supplied in this ZIP.
This review did not run model inference, DirectML, training, or held-out episodes.

## What the code establishes

The loader freezes the admitted brain and the parent executive. Training and
evaluation compare protected tensors with captured references, rather than only
printing preservation claims. The new learned state is a 128-dimensional
recurrent progress controller plus a scalar intent gain. The parent action-intent
head is loaded separately and frozen. The world definition has the same SHA-256
as the previously inspected v3 source.

`train_recurrent_option_progress_v1.protected_sequence` and the equivalent
evaluation function advance the environment with `env.step(teacher)`. The
executive predicts actions at those states, but its predictions do not affect
the subsequent trajectory. At the response state, collection stops and response
logits are scored separately. The reported six-way score is a conjunction of six
response classifications per condition; it is not episode completion.

The donor test is meaningful within this interface: recipient base logits and
recipient execution gate remain fixed while the executive receives the opposite
cue's recalled content. Conversely, ZERO zeros only the executive's recalled
input. Base logits and the progress state still come from the ordinary
memory-enabled trajectory. This is an executive-input ablation, not removal of
memory from the whole agent. OFF/ZERO percentages are paired-cue scores, so 0%
paired success does not imply 0% accuracy on individual responses.

## Why the timing numbers are compatible

The implementation always computes:

```text
final_logits = base_logits + intent_gain * execution_gate * raw_intent
```

The centered sigmoid produces a signed gate between -1 and 1. There is no
runtime threshold at 0.5. The reported pre-response "gate off" rate merely counts
`abs(gate) < 0.5`. A gate of 0.2 still applies 20% of the raw intent, before the
gain factor, and may change the winning action. Thus the logged 97.21% gate-off
rate and 76.68% pre-response action accuracy are not contradictory.

The field named `pre_response_action_preservation` measures whether the result
is UP, rather than whether it equals the base policy's action. It is a useful
task requirement, but the report does not separately reveal base-policy errors
and executive-induced errors at these states.

Timing MSE averages all active steps, including just one positive response
target per sequence. Action losses separately reward correct response logits.
This creates different incentives from the gate admission thresholds. That is a
possible contributor, not a demonstrated sole cause. The results establish
failure of this setup within its budget, not impossibility of recurrent timing.
The stopping rule remains appropriate; this finding is not a recommendation to
start another gate variant.

## Reproduced reporting defects

`reproduce_review_findings.py` exercises the supplied decision script and shell
runner using only the Python standard library and Bash. Runner checks use a
temporary project directory, a simulated checkpoint marker, and stubbed model
commands. No real training is performed.

| Check | Observed result | Consequence |
| --- | --- | --- |
| Perfect synthetic summary, zero underlying records, empty factor table | Decision exits 0 and reports PASS | Admission trusts summaries without validating evidence coverage |
| Nonzero RULE_NONE difference of 0.0000005 | `none_exact_noop` passes | The advertised exact check actually allows a 0.000001 tolerance |
| Evaluator deliberately exits 23, no old report | Runner requests simulated training to 3200 and 4800 | A software error is treated like a learning failure |
| Evaluator deliberately exits 23, old passing report present | Runner exits 0 and reports PASS | Stale evidence can produce false convergence |

The shell issue arises because `evaluate_gate` is called as an `if` condition;
Bash's `set -e` does not provide the expected protection within that function.
The decision script's final exit status can mask earlier evaluation failures.

The evaluator also discards its per-condition rows when saving JSON and omits
the evaluated checkpoint hash and source hashes from that JSON. A future
admission system needs saved records, coverage and arithmetic checks, checkpoint
and source identities, and explicit error propagation. It should distinguish
the expected gate-failure exit code from all software errors and prevent reuse
of stale output.

**These reproductions do not show that David's supplied run falsely failed.**
The log shows completed evaluations and explicit metric failures at each stage.
They identify defects to exclude from the next build's evaluation pipeline.

## Preserve the reusable assets

The v0.19 checkpoint saves the progress controller and gain; it does not embed
the protected memory or the frozen v0.18 action-intent weights. Keep the separate
parents available:

| Asset | Project path | Identity recorded by the supplied code/log |
| --- | --- | --- |
| Frozen v0.18 intent parent | `state/option_intent_executive_v1/latest.portable.pt` | SHA-256 `041acf1bcdf46736a37c12c343cde2d9316b65cf4b4ee08604cc9c572585e5e2` in the run log |
| Admitted memory/recall | `state/working_memory_recall_capacity_control_phase1_v1/latest.portable.pt` | Expected SHA-256 `f9976894357b795525ce464c4ff5ef5a882f00073ed81444184c1bc79756282c` |
| Historical brain | `state/child_v113_dml_smoke/infant_training.pt` | Expected SHA-256 `086932388a9a410af46fea8e87cc63dd4b60161e03e247bf81fc88d28cfd577f` |

These are recorded identities, not hashes independently recomputed from the
trained files during this review.

## Requirements for the planning transition

The recovered `planning_v103.py` uses `PlanningCortex.step` for option-policy
imagination and `OptionCortex` for selection, primitive actions, termination,
and terminal values. It does not invoke `WorldModelEnsemble`. The selector takes
a planning latent without explicit goal or canonical-recall arguments. Adding
memory and goals requires a defined interface, not merely enabling these classes.

The next audit should identify the qualified checkpoint, its planning/spatial
representation and option identities, and matching historical behavioral
evidence. Copying an option head into a different latent space is not justified
by matching tensor shapes.

The reported historical comparison is 38% hierarchical completion versus 61%
direct-option completion and 0% reactive completion. Its original P4b gate
compares planning with reactive control; it does not require planning to beat
direct options. The next experiment therefore needs paired autonomous direct
options and hierarchical planning, with matched memory/goal inputs, plus
memory/goal interventions and prediction-error measurements.

**Next input:** the v0.19.1 source ZIP or its generated
`state/planning_option_integration_audit_v1/audit_bundle.zip`. No additional
v0.19 training or fresh-v4 evaluation is needed to obtain this evidence.

## Reproduction

Extract this audit ZIP into a separate directory. Do not install it over the
project. From its `continuous_brain_v0190_source_audit` directory run:

```bash
python reproduce_review_findings.py
```

The script asserts the observed defects and writes `reproduced_findings.json`.
An assertion passing here means the defect was reproduced, not that the model
passed a capability test. Seven supplied Python files compiled successfully,
and `bash -n` accepted the runner. Source bytes are retained under `source/`.
