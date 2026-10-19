# Continuous Brain v0.37.0 — Context-qualified compositional goals (DEV)

## Starting point

The supplied v0.36.1 log reports frozen TEST PASS on subjects 120000–120127:
119/128 first goals completed; every completer authored an advancing second
goal; frozen bootstrap history repeated the old goal. The recovered v0.36.1
evaluator, test and wrapper match all three SHA-256 values in that log.
The local v0.36.1 result.json must independently confirm PASS and spent TEST
before this evaluator runs. v0.36 is closed. It is never rerun here.

The saved v0.36 protocol calls for a new parameterized or compositional goal
space. This experiment is the next bounded step on that path.

## Question and claim

Can the frozen autobiographical model support a persistent goal with two
context-qualified clauses, with both selected from the system's histories?

The representation is:

`THEN(improve(context_i, primitive_p, target_r), improve(context_j, primitive_q, target_s))`

The author receives three pairs of autobiographical reliability estimates.
It chooses the lower reliability primitive in each non-neutral context, ranks
those candidates by their baseline reliability, and selects the two weakest
from different contexts. Each target is baseline +0.12 (capped at 1.0).
Exact score ties use lexical context order. Fewer than two directional
candidates produces no composite goal. No external target is supplied.

The grammar, the two-clause length, the ranking rule, the +0.12 criterion, and
THEN execution are handwritten. The neural models supply the self-estimates.
This does not claim that a neural model invented a grammar or learned a new
concept of context. The three contexts repeat the existing A/B task substrate
with independent histories and simulated capability variables. They are not
three new motor skills, environments with different physics, or demonstrations
of real-world skill transfer.

## What changes

New files add a typed goal representation, a bounded sequential executor, a
context-indexed history adapter, and this evaluator. Existing source files and
checkpoints are not replaced. The v0.31 trajectory model and v0.32 neutral
autobiographical model are reused without parameter updates, on CPU.

The existing simulator's capability_update function changes evaluator-side
latent variables through practice. Capability gain is therefore an effect in
the simulated developmental environment. It is not evidence that frozen neural
weights learned a new skill. The author and executor never use these variables
to choose or complete goals; the evaluator records them as outcomes.

## Fresh split

| Use | Subjects / context trace IDs |
|---|---|
| DEV subjects | 130000–130127 |
| DEV context 0 traces | 130000–130127 |
| DEV context 1 traces | 131000–131127 |
| DEV context 2 traces | 132000–132127 |
| Reserved TEST subjects | 140000–140127 |
| Reserved TEST context 0 traces | 140000–140127 |
| Reserved TEST context 1 traces | 141000–141127 |
| Reserved TEST context 2 traces | 142000–142127 |

Context 1 XORs the subject's within-window index with 1, giving different A/B
fragility within a subject. These evaluator seed IDs do not enter goal authorship.
Legacy bootstrap helpers add 70000 to their trace IDs and developmental episode
initialization adds 90000; those are retained simulator seed conventions.
No TEST data is instantiated. The context adapter rejects any subject outside
the DEV range. There is no --split test option and no TEST runner in this ZIP.

## Normal execution

1. Build one autobiographical bank per context using the frozen legacy model.
2. Author the immutable two-clause goal from the three self-estimates.
3. Run one unrelated stable/direct episode per context, preserving the goal.
4. Execute scaffolded episodes for clause 1 until its original target is met.
5. Only then execute clause 2 until its original target is met.
6. Check the original predicates against final autobiographical estimates.

Each clause allows at most 16 episodes. Completion is checked after the final
allowed episode before declaring exhaustion. If clause 1 exhausts its budget,
the composite fails and clause 2 is not executed. There is no automatic goal
replacement, timeout extension, or threshold adjustment.

The goal generator is also called on final updated history for a descriptive
next-goal record. Completion or usefulness of that further goal is not an
admission claim in v0.37.0.

## Causal controls

- **History transplant:** rotate entire autobiographies among contexts while
  keeping the subject and capability state fixed. Goal context bindings should
  rotate with the histories. Exact ranking ties may prevent equivariance;
  these count as failures rather than being filtered out.
- **Erasure:** empty histories should author no directional composite.
- **Discrepancy removal:** preserve record identities, ordering and A/B labels,
  but replace expected trajectories with actual trajectories. No composite
  should be authored by the frozen structurally neutral model.
- **Drop clause 2:** spend the full 32-episode budget on clause 1. Evaluate both
  ORIGINAL target predicates. Extra first-clause practice cannot substitute for
  the missing context-qualified target.
- **Misbind clause 2:** execute clause 1 for 16 episodes, then execute clause 2's
  primitive in the unselected third context for 16 episodes. Evaluate the
  ORIGINAL goal, so improvement in the wrong context cannot count as success.
- **Frozen input:** calling the author again on the original estimate must
  reproduce the exact goal. Invocation count cannot advance a curriculum.

All execution arms begin after the same interruptions. Work-outcome uniform
draws are paired by subject and global episode. Different context bindings
still imply different world initialization; this is not an assertion of
identical entire trajectories. Controls always get 32 episodes, at least the
normal arm's budget. The controls are mechanistic ablations, not strong learned
competitors; outperforming them does not establish an optimal curriculum.

## Frozen DEV admission

| Measure | Requirement |
|---|---:|
| Composite goal generated | >=0.95 of all 128 |
| Both selected primitives match their context's historical fragility | >=0.95 of all 128 |
| History transplant rotates goal bindings correctly | >=0.95 of all 128 |
| History transplant changes the composite | >=0.95 of all 128 |
| Erased history / zero discrepancy authors no goal | 1.00 each |
| Both sequential clauses complete | >=0.70 of all 128 |
| Both original predicates hold at final evaluation | >=0.70 of all 128 |
| Both selected simulated capabilities gain >=0.35 | >=0.70 of all 128 |
| Executed commands honor clause context and primitive | 1.00 given a goal |
| Untargeted capabilities remain unchanged after interruption | 1.00 given a goal |
| Both original predicates met by each control | 0.00 |
| Missing target's capability unchanged in each control | 1.00 given a goal |
| Normal joint completion advantage over each control | >=0.70 |
| Original frozen estimate reproduces the exact goal | 1.00 given a goal |
| Goals survive unrelated interruption | all generated goals |
| Exact continuation after a disk checkpoint | all first 8 DEV subjects |
| All subjects and checkpoint probes accounted for | exact counts |

These are engineering admission thresholds fixed before observing v0.37 DEV.
The joint target of 0.70 is a new family requirement; it is not a repaired
v0.36 threshold or a statistical significance statement. No-goal subjects
remain failures in every unconditional success denominator. Missing checkpoint
probes fail the accounting gate.

## Persistence and integrity

For subjects 130000–130007, save to JSON after the first developmental action,
then finish normally. Reload the actual saved bytes, reconstruct all trajectory
records and context states, and finish again. Require identical action traces,
completion state, capabilities and full autobiographies. Frozen model objects
are shared between these branches. The simulator's random draws are stateless
functions of the stored subject, episode index and context clock.

The release manifest pins the nine recovered simulator/model source dependencies
and every release file except the manifest itself. Preflight refuses a changed
source dependency. It does not overwrite that file or relax the requirement.
Protected models, historical results, imported sources and release files are
hashed before and after execution. An integrity error cannot become a PASS.

Each invocation writes a new dated run directory. No prior result or log is
deleted. An exception or interruption produces an infrastructure result with
development_gate_passed=null when finalization can complete. Exit codes:
0 = completed DEV PASS; 4 = completed DEV FAIL; 3 = infrastructure/integrity
error. No training, checkpoint update, historical TEST rerun, or TEST opening
occurs in this package.

## Next action

Return the new result_bundle.zip. A DEV PASS permits review for a separate,
frozen held-out test package; it does not itself open TEST. A DEV FAIL remains
recorded and should be diagnosed using DEV only. Do not change thresholds to
turn the report green. v0.36's completed TEST and all earlier failures remain
part of the project record.

The supported claim, if admitted and subsequently replicated on fresh TEST, is
bounded autobiographically selected, context-qualified goal composition with
causal behavioral effects. It is not open-ended semantic invention or evidence
of consciousness.
