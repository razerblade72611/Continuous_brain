# Continuous Brain v0.22.0 — Additive Goal-Diverse Temporal Options

## Why this branch is warranted

v0.21.0 established that admitted memory is not the failed component:
- TRAIN remembered-goal decoding: 100%
- DEV remembered-goal decoding: 100%
- donor memory -> donor goal: 100%
- zero memory -> exact selector baseline: 100%

But the historical skill bank could not reliably express the remembered goal.

v0.21.1 enumerated all six frozen historical options on the spent DEV states:
- any immediate correct-goal option: 48.4375%
- any temporally correct-goal option: 66.0156%
- both LEFT and RIGHT temporally supported in a paired condition: 32.03125%
- any correct-completion option: 14.0625%
- both sides correct-completion support: 0%

Therefore another selector over the same six options is not warranted.

## Preservation strategy

Do not resize or overwrite historical `OptionCortex`.

The historical six are checkpoint tensors whose shapes depend on
`option_count=6`.  v0.22 adds a separate sidecar bank containing two new
goal-conditioned temporal skills while preserving all six historical options
bit-exact.

## New temporal skills

`VariedCuedDetourWorld` provides a clean option target.

After the delayed decision the teacher executes a variable-duration LEFT or
RIGHT traverse.  When the correct horizontal displacement has been achieved,
the teacher switches to vertical/idle motion.

The new bank therefore learns:

remembered goal
 -> choose LEFT or RIGHT skill
 -> execute lateral action for a variable number of steps
 -> learn termination at the end of the lateral traverse
 -> hand control back to the frozen historical direct-option controller

Architecture:
- fixed two-state goal embedding;
- frozen PlanningCortex latent input;
- shared MLP;
- primitive-action policy head;
- termination head.

Only this new sidecar bank is optimizer-owned.

## Training

TRAIN conditions: 0..1023, both LEFT and RIGHT members.

The cue-independent prefix is teacher-forced only to reach the delayed skill
initiation state.

For every lateral teacher state:
- action target = LEFT or RIGHT;
- termination target = 0.

At the first state after the lateral traverse, where teacher action becomes
non-lateral:
- termination target = 1;
- no policy target is applied.

The admitted memory, historical options, world model and planner are not
trained.

## Fresh DEV evaluation

DEV conditions 1728..1855 are fresh relative to v0.21.0/v0.21.1.
TEST 1980..2359 remains untouched.

At each delayed decision:
1. admitted working memory carries the vanished cue;
2. the frozen v0.21 goal decoder creates the explicit goal;
3. the new corresponding goal option owns primitive control;
4. learned termination hands control back to the historical direct-option bank.

Causal controls:
- opposite-cue donor memory;
- explicit goal flip;
- historical direct-option baseline.

Frozen DEV gate:
- remembered goal accuracy >= .95
- own goal -> correct lateral commitment >= .95
- both LEFT and RIGHT correct in >= .90 paired conditions
- donor goal -> donor-directed commitment >= .95
- explicit goal flip -> flipped commitment >= .95
- learned option terminates >= .90
- termination duration within +/-2 teacher lateral steps >= .85
- commitment advantage over historical baseline >= .25
- historical baseline commitment <= .70

Full correct-goal completion is reported but is secondary in this milestone.

## Meaning of PASS

A PASS establishes this integrated functional chain on held-out DEV conditions:

past observation
 -> admitted working memory
 -> canonical recalled content
 -> explicit remembered goal
 -> learned goal-diverse temporal skill
 -> learned duration/termination
 -> autonomous primitive behavior
 -> handoff to older skills

It does not establish consciousness, broad general intelligence, or general
episodic memory.

After PASS the next architectural step is to let the remembered goal and the
expanded skill bank participate in consequence-based planning/replanning, then
move to multi-item memory and richer 3-D embodiment.

## Meaning of FAIL

Do not reopen the old six-option selector or Detour planner loops.

Use the failed gate to isolate one of:
- new skill policy learning;
- temporal termination learning;
- memory-to-skill invocation;
- historical post-skill handoff.

Repair only that integration layer.
