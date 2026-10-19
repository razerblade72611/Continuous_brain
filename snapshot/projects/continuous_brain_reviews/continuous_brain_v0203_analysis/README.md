# Continuous Brain v0.20.3 — completed results analysis

This package analyzes the supplied results. It is not a new brain release and
does not install or change a controller. No training, neural inference, or
checkpoint access was performed. The analysis replays only the already-recorded
actions on the already-used development seeds.

## Decision

Preserve selector support and the direct-option baseline. The result supports
the selector's practical value, but does not establish that softening its
support will improve autonomous completion. The next experiment should test
the accuracy and decision value of the existing planner's predictions under
full-episode continuation, before changing its authority over execution.

The broader goal remains a persistent learning agent that uses memory, goals,
prediction, and feedback to improve multi-step behavior. The next milestone is
a reliable benefit from deliberation over its own direct learned skills.
Memory integration remains paused and the v0.19 timing branch remains closed.

## What was verified

- Recomputed every published aggregate field from all 480 option rows.
- Reproduced every recorded winning option using the original lexicographic
  ordering: completion, fewer collisions, then segment reward. Exact ties favor
  the earlier selector rank because rows are already sorted by selector rank.
- Reconstructed all 80 starting states from seeds and previously recorded
  selector actions; no action-history gap occurred between recorded boundaries.
- Replayed all 480 recorded segments in the existing DetourWorld source.
  Completion, collision counts, and segment lengths matched. The maximum
  absolute reward error was exactly zero in this runtime.
- Verified rejection of empty evidence, a stale/inconsistent summary, and
  recorded actions that disagree with reported outcomes.

This verifies arithmetic and recorded environment outcomes. It does not
independently verify the original neural policy, the original checkpoint hashes,
or the quality of any imagined predictions. Those predictions are absent from
this results archive.

## The 63.75% result, unpacked

The reported outside-top-two winner rate is 51/80. Comparing the best outside
option against the best top-two option at each boundary gives:

| Decisive reason for an outside win | Boundaries |
| --- | ---: |
| Completion available only outside top two | 0 |
| Fewer collisions, with completion status tied | 0 |
| Higher segment reward, with completion and collisions tied | 51 |

Both groups contained at least one collision-free segment at every boundary.
Of the 51 outside winning segments, 50 did not complete the task. The remaining
one completed in three actions where the best top-two option completed in two.
It won on the small distance-shaping reward difference, not on earlier success.
That example is seed 14001, step 26: reward 2.0017887105 for the outside option
versus 2.0007272197 for the top-two option.

Across all 80 boundaries, completion coverage was:

| Completion somewhere in each group's recorded segments | Boundaries |
| --- | ---: |
| Top two only | 6 |
| Both groups | 4 |
| Outside only | 0 |
| Neither group | 70 |

Thus, outside support added zero unique within-segment completion opportunities
in this sample. This is not proof that outside options never improve eventual
completion after further actions.

The median reward advantage among the 51 reported outside wins was
0.0021955049. Thirty-two were smaller than the environment's 0.004 per-action
cost in magnitude. One was only 8.67e-19, a numerical tie at ordinary floating
point tolerance. Comparing the magnitude with a step cost is descriptive, not
a proposed override threshold.

## Duration matters, in both directions

My earlier caution mentioned short segments winning by stopping early. The
actual rows show a broader duration effect: 34 outside winners ran longer,
11 had equal length, and six ran shorter than the best top-two segment.

Twenty of the longer winners were simply extensions of the top-two option's
identical action prefix. Five shorter winners were identical prefixes of the
longer top-two segment. Different option IDs therefore often describe shared
primitive behavior over the part being compared.

For each of the original 51 winning pairs, I replayed both saved sequences only
through the shorter sequence's length. This comparison preserves the same
starting state and equal numbers of actions. It does not invent actions after
an option terminates.

| Comparison of the original 51 winning pairs at equal prefix length | Original floating point ordering | Treat reward differences <= 1e-12 as ties |
| --- | ---: | ---: |
| Outside still wins | 17 | 15 |
| Tie | 27 | 30 |
| Top two wins | 7 | 6 |

The tolerance removes numerical ordering artifacts; it is not a learned
parameter or policy threshold. Completion and collision differences always
take priority.

This sensitivity check shows that the 51/80 headline depends substantially on
comparing different stopping times. It does not show that every longer option
is bad or that the 15 surviving prefix wins improve full-episode performance.

As a separate sensitivity check, re-selecting the best option in each group
using the first three recorded actions gives 35 outside wins, 37 ties, and
eight top-two wins with that numerical tolerance. This uses different
comparisons from the original 51 pairs. It remains a very short-horizon test,
with four outside candidates competing against two inside candidates, and is
not a new planning success measure.

## Where collision exposure is concentrated

| Selector rank | Segments | Completions | Total collisions | Mean segment length |
| --- | ---: | ---: | ---: | ---: |
| 1 | 80 | 9 | 16 | 5.3125 |
| 2 | 80 | 4 | 12 | 8.2750 |
| 3 | 80 | 2 | 12 | 9.2375 |
| 4 | 80 | 1 | 0 | 4.4250 |
| 5 | 80 | 1 | 109 | 9.0625 |
| 6 | 80 | 0 | 195 | 9.1750 |

Ranks five and six account for 304 of the 316 outside-group collisions (96.2%).
Rank four had no observed collisions. The outside group is therefore not
uniformly dangerous, and rank should not be treated as a calibrated risk scale.

Duration alone does not explain the aggregate collision gap: top-two segments
had 2.58 collisions per 100 recorded actions, versus 12.38 outside. These are
pooled descriptive rates on selected trajectories, not independent collision
probability estimates or a deployment guarantee.

## Limits

- The sample comprises 80 clustered boundaries from seeds 14001–14015. The
  diagnostic stopped at its boundary cap before evaluating all 40 advertised
  development seeds. The final sampled episode was truncated by that cap.
- The states were encountered by the direct-option controller. They do not
  cover all states visited by a planner that repeatedly overrides it.
- The six alternatives at a boundary share a starting state; the 480 rows are
  not 480 independent episodes. Candidate-count and trajectory-clustering
  effects remain relevant.
- Recorded segments last at most 12 actions. The data do not include each
  counterfactual branch's subsequent closed-loop continuation or the planner's
  imagined scores. Neither can be recovered from these outcome averages.
- Counterfactual winners use privileged knowledge of realized outcomes. They
  establish opportunities under a ranking, not a deployable prediction rule.
- The run and this analysis are software/measurement checks. Neither is a
  generalization result or evidence that the new planner improves control.

## Next experiment

See `NEXT_EXPERIMENT.md` for one bounded test that separates availability of
better options from the planner's ability to identify them before acting.
Do not promote outside options solely because this diagnostic recommends it.

## Reproduce this completed analysis

No installation into the Continuous Brain project is needed. Extract this
archive anywhere, activate an existing Python environment with NumPy, and run:

```bash
python analyze_v0203.py
```

The command recreates `analysis.json` and `boundary_comparisons.csv` from the
included unchanged input archive and environment snapshot. It loads no neural
checkpoint. NumPy is its only third-party dependency.

Files include:

- `analysis.json`: complete recomputation, sensitivity counts, provenance, and
  80 paired boundary comparisons.
- `boundary_comparisons.csv`: the same paired comparisons in tabular form.
- `analyze_v0203.py`: reproducible analysis with evidence validation.
- `validation.json`: positive replay and negative input checks.
- `source/environment.py`: unchanged environment source used for replay.
- `source/diagnose_option_support_counterfactual_v1.py`: the original diagnostic
  for inspecting its scoring logic; do not run it from this analysis package.
- `input/continuous_brain_v0203_results.zip`: the unchanged supplied results.
- `MANIFEST_SHA256.json`: hashes of the packaged files, excluding the manifest
  itself.

Input archive SHA-256:
`096e675fdabefe1105e6ba5532aa11ac0d5032f8cade2ebd517226c649a47009`
