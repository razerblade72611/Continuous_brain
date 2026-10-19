# Continuous Brain v0.7.0 — Developmental semantic architecture

## One individual, several neural subsystems

v0.7 keeps the v0.6.2 persistent executive and learned motor.  It adds a separate
semantic/communication cortex that can be trained independently, qualified, and
then attached to the same persistent identity.

```text
                     DEVELOPMENT ONLY

      frozen DINOv3                 frozen Qwen3.8 teacher
           |                                  |
           | visual embedding                 | structured labels
           |                                  | Q/A, concepts, care
           +------------------+---------------+
                              v
                  trainable semantic cortex
                 +--------------------------+
                 | visual adapter           |
                 | concept head             |
                 | question encoder         |
                 | answer decoder           |
                 | care/social head         |
                 +------------+-------------+
                              |
                        qualification
                              |
                              v
                 attached to same individual

                     QUALIFIED RUNTIME

 egocentric RGB -> frozen DINOv3 -> semantic cortex -> native short answer
                                      ^
                                      |
                    sensed/self/goal/memory context

                           Qwen absent
```

The long-lived executive remains:

```text
perception + self + intention + episodic retrieval
                    |
                    v
           recurrent workspace
                    |
          option values/world model
                    |
                    v
             learned motor/body
```

v0.7 deliberately does not route Qwen output into motor control or overwrite the
persistent workspace.  The teacher creates supervised targets; it does not act as
a hidden executive.

## Why a separate cortex first

The existing executive expects 54 sensory values (six structured plus 48 visual).
Changing this tensor shape would force a risky migration of 32M learned parameters.
Instead, v0.7 trains semantic capability beside the executive first.  The semantic
visual adapter also learns a compatibility target toward the existing 48-value
pixel-statistics representation, creating a future path for gradually replacing
the primitive visual code without destroying the individual's existing tensors.

This is developmental scaffolding, not a permanent prohibition on tighter
integration.  Later releases can feed the learned semantic code into the recurrent
workspace once its behavior and retention are measured.

## Communication context contract

Exactly 48 numeric values enter the semantic student alongside the DINO embedding:

- 6 current structured sensor values;
- 24 empirical self-model values;
- 4 one-hot goal values;
- 6 one-hot last-skill values;
- 8 latest-memory summary values (six skill bits, reward, success).

No world coordinates, object IDs, top-down camera, Qwen hidden state, teacher answer,
or simulator diagnostics are accepted by the neural communication path.

## Care as learned consequence sensitivity

The first care head predicts six bounded quantities:

`social_presence`, `help_needed`, `harm_risk`, `respect_request`, `uncertainty`,
`care_priority`.

These are training targets designed to make social consequences visible to later
decision learning.  They should eventually be connected to the world model and
policy-value system rather than remaining a language-only label.  They are not
claimed emotions.

## Development curriculum

### Stage A — deterministic nursery

Use DevelopmentRoom to validate collection, DINO caching, teacher labels, training,
resume behavior, qualification and basic conversation mechanics.

### Stage B — 3D semantic nursery

Use several AI2-THOR scenes and many egocentric poses.  The student sees only image
and permitted internal context.  Simulator object metadata may be retained as an
evaluator-only field to score labels; it is never fed to Qwen or the student.

### Stage C — frozen-teacher distillation

Qwen sees each camera frame and permitted context description.  It returns a short
literal summary, concepts from a fixed vocabulary, grounded Q/A pairs, and bounded
care/uncertainty targets.  DINO produces frozen visual vectors.  Only the student
optimizer changes.

### Stage D — teacher-free qualification

Held-out records test native answer-token accuracy, concept F1, care error, visual
compatibility and autoregressive generated-answer similarity.  Only a passing model
becomes `qualified_semantic.pt`.

### Stage E — attachment and first conversation

The qualified snapshot is attached to a continuation copy of the persistent
individual.  The same UUID/history is preserved.  `continuous-semantic chat` then
uses DINO + the attached student; Qwen is not called.

## Imagination policy

The v0.6 executive can already predict short sensory/reward consequences.  Rich
semantic imagination should be added progressively:

1. train on real episodes;
2. measure one-step and multi-step calibration;
3. generate counterfactuals from a real starting state;
4. mark every synthetic item `teacher_counterfactual` or `model_imagined`;
5. weight imagined items below real items;
6. never write imagined events into autobiographical memory as lived facts;
7. compare imagined predictions with later real outcomes and learn the error.

This lets imagination accelerate learning without confusing memory, fantasy and
experience.
