# Option continuation and prediction diagnostic v1

This is a runnable diagnostic, not a trained brain update. It loads the exact
canonical 14K checkpoint strictly, uses CPU inference, and preserves the
existing learned selector, action policies, termination policy, memory, and
executive checkpoints. It installs no replacement `continuous_brain` modules.

## What it tests

At each of the same 80 saved decision boundaries from v0.20.3:

1. Capture the existing historical hierarchical planner's choices and imagined
   option trajectories before executing any real counterfactual branch.
2. Execute each of the six initial options until its actual learned termination
   or episode end. A 12-step diagnostic/imagination budget does not become an
   execution timeout.
3. After the first option terminates, follow the same frozen direct selector
   and option policies, responding to each branch's actual observations.
4. Stop at success or the original absolute episode deadline of 120 steps.
5. Check the initial action sequence, reward, collisions, completion, and
   termination against the original saved v0.20.3 segment.
6. Compare predictions made from observed states with predictions rolled
   forward under the same executed actions, over the initial option's first
   12 steps at most. Autonomous imagined option actions are also recorded.

The saved states are embedded in `option_continuation_prediction_v1/input.json`
with an exact expected hash. These are the already-spent seeds 14001–14015;
no new environment seed window is used.

The comparators are direct selection, the existing `planning_v103` hierarchical
planner with top-two support, the same planner with all-six support, and two
oracles selecting the best realized initial option within each support set.
After the initial intervention, all branches use direct control. Oracles use
future outcomes solely for diagnosis and are not deployable policies.

**Scope:** the planner comparators are the recovered historical implementation:
option depth 2, beam width 8, imagined option limit 12, discount 0.95, minimum
commitment 3, termination threshold 0.5. The separate v0.20.2 goal-wrapper source
was not supplied; this package does not claim to reproduce that wrapper.
No goal weight is tuned and no memory input is connected.

## Completed author validation

I ran all 480 full continuations locally against a checkpoint whose SHA-256
matches the canonical checkpoint exactly. All 480 saved action/outcome prefixes
matched. Strict loading passed and in-memory model tensors were unchanged.
Core execution took approximately 165 seconds in this environment; your runtime
may differ.

| Initial-choice comparator | Previously failing boundary continuations rescued | Previously successful boundary continuations broken |
| --- | ---: | ---: |
| Historical planner, top two | 1 | 4 |
| Historical planner, all six | 3 | 10 |
| Best realized top-two choice, using future information | 5 | 0 |
| Best realized all-six choice, using future information | 10 | 0 |

These are paired boundary counts from 80 states clustered within 15 spent
development episodes. They are not independent episode counts, fresh test
results, or success rates for a planner repeatedly controlling a whole episode.
The report includes exploratory paired summaries resampled by seed.

Better initial choices exist even inside the supported set. The historical
planner failed to capture their benefit reliably in this sample. That supports
repairing prediction/ranking and execution semantics before expanding control
authority or reconnecting memory.

## A concrete mismatch exposed by this test

The historical `_rollout_option` ends its computation at a 12-step budget even
when its `terminated` flag is false. `_hierarchical_option_choice` nevertheless
expands the next imagined option. Real execution only releases an active option
at learned termination; it has no corresponding 12-step timeout.

For the all-six comparator, all ten completion losses had an initial imagined
option cut off at 12 steps without predicted termination. In real execution,
those initial options retained control for 103–120 steps and ended only when
the episode did. Eight of those ten cases had 102–113 remaining collisions;
two failed without collisions.

Example: seed 14003, decision step 1. The planner chose option 5 (selector rank
5), imagined 12 steps without termination, and then planned a different option.
In reality, option 5 retained control for all 120 steps, caused 113 collisions,
and failed. Direct control completed that episode.

This is a source-level contract mismatch supported by the recorded cases.
Its association with every all-six completion loss in this sample does not
establish that correcting it alone will solve planning; collision prediction
and terminal scoring can also be wrong.

The first repair target is therefore: **a computation budget limit must not be
treated as permission to switch an active option.** At a truncated imagined
option, preserve its identity/age and model continued control, or use an
explicitly justified value estimate conditioned on continuing that option.
Do not silently add an execution timeout to make the old imagination appear
correct. Any intentional change to interruption policy is a separate controller
change requiring its own comparison.

## What the local validation covered

- All 80 boundaries and 480 continuations with exact saved-prefix agreement.
- Strict canonical checkpoint loading and exact in-memory tensor preservation.
- Direct continuation consistency across all recorded boundaries of each seed.
- Independent comparison with the original `_run_option_episode` on seeds
  14001 and 14002, matching success, terminal step, reward, and collisions.
- Five execution-contract tests covering natural termination, continued option
  control beyond 12 steps, the absolute deadline, branch isolation, and rejection
  of missing evidence.
- Rejection of duplicated boundary evidence, a comparator outcome unsupported
  by its selected branch, and an incorrect episode deadline.
- Wrapper failure behavior with a simulated evaluator crash and an old report
  present: nonzero exit, no COMPLETE marker, old report untouched.

The isolated author run had the canonical checkpoint, but not all four
protected user-project checkpoints. It tested the complete inference core and
saved raw evidence; it is not represented as a successful run of the user
project's four-file preflight. Your normal wrapper requires and verifies all
four checkpoint hashes before and after execution.

Author evidence is included in
`option_continuation_prediction_v1/author_validation/`.

## Install and run in WSL

Download `continuous_brain_option_continuation_prediction_v1.zip` to Windows
Downloads, then run:

```bash
(
  set -euo pipefail
  cd "$HOME/projects/continuous_brain_v0.12.2"
  source "$HOME/.venvs/continuous_brain_v110_dml/bin/activate"
  unzip -o \
    /mnt/c/Users/davei/Downloads/continuous_brain_option_continuation_prediction_v1.zip \
    -d .
  python option_continuation_prediction_v1/test_execution_contract.py
  bash run_option_continuation_prediction_v1.sh
)
```

The existing environment already provides PyTorch and NumPy. Do not install the
author's isolated test-runtime dependencies into your environment.

Each run creates a new directory under
`state/option_continuation_prediction_v1/run_XXXXXXXX/`, containing:

- `report.json`: validation, provenance, paired summaries, prediction errors,
  and imagination/execution duration diagnostics.
- `boundaries.jsonl`: all per-boundary imagined predictions, real actions, and
  outcomes; the summary is independently recomputed from this file.
- `run.log`: progress and errors.
- `result_bundle.zip`: the three files above, created only after verification.

Expected successful markers include:

```text
PROTECTED_UNCHANGED: true
MODEL_TENSORS_UNCHANGED: true
SAVED_PREFIXES_MATCHED: 480
OPTION_CONTINUATION_DIAGNOSTIC_COMPLETE
REPORT_RECOMPUTED: true
OPTION_CONTINUATION_WRAPPER: COMPLETE
```

`COMPLETE` describes completion of the diagnostic, not a planning capability
pass. The final `Upload:` line gives the exact results ZIP path. A provenance,
prefix, or budget mismatch stops the run. Existing source/checkpoint mismatches
must be investigated rather than bypassed.

## Interpretation and next decision

Reproduce the continuation evidence, then repair the imagination/execution
contract in an isolated candidate. Retain direct control and top-two support as
the default comparison. Measure whether that repair captures available gains
without creating new failures; do not infer success from removing the mismatch
alone. Freeze any candidate and evaluation criteria before a subsequent fresh
held-out comparison. The v0.19 timing branch remains closed.
