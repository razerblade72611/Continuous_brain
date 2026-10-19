# Continuous Brain — Next Intelligence Phase after Working-Memory DEV

The project should stop treating device/backend compatibility as the scientific
objective. DirectML remains an execution backend and regression requirement only.

## If DEV passes: freeze and promote reusable working memory

Promote:
- the 64-D gated writer;
- the parameter-matched ungated nonlinear memory→workspace reader;
- preservation checks for the historical substrate.

Archive as controls:
- M1 writer-only;
- M1.5 minimal linear reader;
- M2 contextual gated reader.

Do not add another recall gate variant.

## Next scientific milestone: task-conditioned executive composition

Build an executive module that lets the brain itself—not a post-hoc probe—use
the same remembered content differently according to goals/instructions arriving
after storage.

Training families should require several operations over memory, for example:
- NORMAL / INVERT;
- MATCH / NONMATCH;
- choose first vs. second remembered item;
- delayed comparison;
- conditional action based on current goal;
- short multi-item sequence recall.

Critical design rule:
the memory writer is shared. Task meaning is not baked into the stored item.

Success requires held-out compositions, not only held-out coordinates.

## Then: model-based planning

Use the existing world predictor as an internal predictive substrate:
- generate candidate short action sequences;
- roll them through the world model;
- score predicted consequences against current goal;
- execute/replan;
- compare against reactive/no-planning controls.

This is where working memory becomes useful for deliberation rather than only
delayed classification.

## Add metacognitive confidence as an additive mechanism

Implement the previously proposed domain-general confidence signal:
- target = whether the brain's own world/action prediction was subsequently
  correct;
- never train it directly from task reward;
- expose it to executive/planning policy as a confidence estimate;
- test whether intervention on confidence changes information gathering,
  memory use, replanning, or abstention.

This adds to the existing architecture; it does not replace it.

## Continual learning / consolidation

Once executive and planning mechanisms work:
- protect the 14k historical substrate and admitted cognitive additions;
- add episodic/semantic retrieval memory;
- consolidate useful experiences;
- require retention gates on old competencies before promotion.

## Embodiment

Move the admitted architecture into AI2-THOR / richer 3-D tasks:
- persistent navigation;
- object permanence;
- multi-room goals;
- remember-now/use-later instructions;
- tool/object interactions;
- long-horizon replanning.

Then progress toward physical robotics after simulator safety/reliability.

## Main-goal framing

These milestones target increasingly general functional intelligence:
persistent memory, flexible recollection, executive control, prediction,
planning, metacognition, continual learning, and embodiment.

They are meaningful functional correlates relevant to the long-term research
goal. None individually establishes subjective consciousness.
