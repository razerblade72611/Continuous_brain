# Continuous Brain v0.13.x — Intelligence Development Sequence

Working memory has passed held-out DEV. Active research now moves upward in the
cognitive stack.

## Milestone E1 — Late-bound executive composition (current)

Make the brain itself combine remembered content with goals that arrive later.

Initial operations:
- NORMAL / INVERT;
- then MATCH / NONMATCH;
- select first vs. second remembered item;
- delayed comparison;
- conditional rule switching;
- short ordered memory sequences.

Goal: one shared memory substrate, many uses. Task semantics must not be baked
into the memory writer.

## Milestone E2 — Multi-item working memory

Extend the currently admitted one-vector memory into a small protected
multi-slot or recurrently addressable working-memory system.

Requirements:
- order sensitivity;
- selective read/write;
- interference tests;
- overwrite/retention controls;
- compositional operations over two or more remembered items.

Do not replace the admitted one-item mechanism until the extension reproduces
its admitted competencies.

## Milestone P1 — Goal-conditioned model-based planning

The project already contains an action-conditioned world model and historical
planning organs.

Next use them as an actual deliberative loop:

1. encode current state + current goal;
2. generate candidate action/option sequences;
3. predict consequences through the world model;
4. score predicted states against the goal;
5. execute the chosen prefix;
6. observe prediction error;
7. replan.

Require controls against:
- reactive policy only;
- random candidate plans;
- planner disabled;
- corrupted world-model predictions.

Planning earns promotion only if intervention on imagined consequences changes
behavior in the predicted causal direction.

## Milestone M1 — Metacognitive confidence

Add the previously proposed domain-general confidence signal.

Training target:
- whether the brain's own action/world prediction subsequently proved correct.

Never train confidence directly from task reward.

Use it to influence:
- whether to gather more information;
- whether to trust/retrieve memory;
- whether to replan;
- whether to abstain or ask for help;
- how much computation to spend.

Causal test:
manipulating confidence while holding first-order evidence fixed must predictably
change these policies.

## Milestone C1 — Continual self-development

Integrate protected promotion and retrieval memory:
- episodic traces;
- semantic consolidation;
- skill memory;
- retention gates;
- rehearsal/replay;
- protected old competencies.

New learning is promoted only if:
- target competence improves;
- admitted prior competencies remain within retention bounds.

## Milestone B1 — Long-horizon 3-D embodiment

Move admitted executive/planning/metacognitive architecture into AI2-THOR:

- remember-now/use-later multi-room goals;
- object permanence;
- delayed instructions;
- route replanning;
- tool/object interactions;
- multi-step household tasks;
- persistent autonomous exploration.

After simulator reliability:
- transfer the same architectural contracts to physical robotics.

## Long-term target

The engineering target is increasingly general, persistent, self-directed
intelligence with:

- reusable memory;
- flexible executive control;
- predictive world modeling;
- deliberative planning;
- metacognitive self-monitoring;
- continual learning;
- embodiment;
- persistent identity/continuity.

These are serious functional capabilities relevant to the project's ultimate
research goal. They do not by themselves establish subjective experience.
