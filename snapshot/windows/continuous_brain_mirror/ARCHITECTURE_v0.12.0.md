# v0.12.0 local developmental architecture

## Scope and lineage

This adds a new, small runtime to the complete v0.11.4 source tree. The original
`InfantBrain` architecture, checkpoint loader, planner/options, and protected
trainer remain available. The new `DevelopmentBrain` reuses the existing
DirectML-safe recurrent cell, but has its own checkpoint schema, sensor contract,
model, identity, and environment. There is no automatic old/new weight merge.

Default agent: **667,130 trainable parameters**. Development forecaster:
**451 parameters**. Active/founder/candidate copies are comparisons of one
architecture, not additional independent intelligence.

## Information flow

| Component | Inputs and function | Learning/control role |
|---|---|---|
| Perception | Four-channel 9×9 grid: walls, food, water, agent | Learned convolutional encoder; explicit full-grid simulated sensing |
| Internal-state encoder | Energy, hydration, integrity, resource beacons, previous action/feedback, timing, position, world kind, behavioral history | Learned mapping to recurrent inputs |
| Episodic memory | Stable sensor keys from lived experiences | Retrieves recent/formative outcomes; all events remain on disk |
| Recurrent workspace | Perception, internal state, retrieved memory | DirectML-safe GRU; drives policy/value and consequence prediction |
| Policy/value | Recurrent state | Five actions; policy logits plus a small bounded Q correction |
| Consequence model | Recurrent state and hypothetical action | Predicts body state, reward, collision, and resource event; proposals remain in shadow |
| Development model | Current measured competence and a proposed update budget | Predicts before/after learning deltas; learns after outcomes are measured |
| Admission controller | Paired closed-loop measurements | Applies the explicit retention/acquisition rule; never trusts an unqualified forecast |

This resource world deliberately provides structured resource beacons and a
full grid. It does not demonstrate visual localization, language grounding, or
physical-world autonomy. The original v0.11 benchmarks use different contracts.

## Positive and negative feedback

The environment owns bounded energy, hydration, and integrity variables. Both
energy and hydration decay while the agent rests. Food replenishes energy;
water replenishes hydration; resting can repair integrity while consuming other
resources. Resources saturate at 1.0. Collisions damage integrity.

Positive feedback includes actual resource/integrity recovery and a small bonus
for a newly visited cell. Negative feedback includes depletion, collisions, and
a per-step cost. Feedback is computed from actual transitions, not from neural
predictions or a writable satisfaction meter. There is no equation equating
negative numbers with felt pain, or prediction error with harm.

`audit` tests a narrow tampering opportunity: editing the observation's reported
resource levels does not change the actual body or its reward. This does not
claim protection against an agent with arbitrary code/filesystem access. The
agent's action interface contains only the five simulated actions.

## Learning and persistence

The starter is initialized from scratch and trained with labelled shortest-path
demonstrations. Two corrective rounds collect states visited by the learner's
own actions and ask the authored teacher for action labels. All teacher access is
training-only. Default live sessions use no live teacher or external model.

Live action selection, environment transitions, memory retrieval, behavioral
history, and recurrent context persist. Bounded recent replay rehearses lived
sequences alongside the fixed demonstration data. Candidate learning combines
action imitation on labelled rows, action-value TD, consequence prediction,
and reference-policy matching on rehearsal. Surprise never opens growth gates
or exponentially increases the learning rate.

Each candidate trains separately. Baseline and candidate evaluations use their
own closed-loop trajectories, own recurrent states, and own within-evaluation
memories. Their episodes are evaluations, not inserted as autobiographical
lived experiences. The hypothetical action proposals are labelled within lived
event records as predictions, not falsely recorded as executed actions.

Admission requires all of the following on each task family:

- At least 95% of the current active policy's success count.
- At least 95% of the fixed founder's success count, preventing gradual loss
  through a chain of locally acceptable replacements.
- Mean return no more than 0.10 below either reference, and no more than 0.5
  additional mean collisions.
- Measured acquisition: higher completion/return or at least 1% lower body
  prediction MSE on one family. A behaviorally unchanged but more accurate
  predictor can qualify; the reason is recorded.

Lost/gained seed outcomes are reported separately. These are finite-sample
rules, not statistical confidence guarantees. The 12 admission episodes are
reused and are therefore development data. The separately reported starter
validation uses new seeds and no model updates; neither suite proves broad
generalization.

After promotion, the current episode's actual input history is replayed through
the new recurrent network to rebuild compatible context. The history remains
intact. Hypothetical branches do not overwrite the lived workspace. SQLite
commits model state and pending events in one transaction; single-writer locks
work on Windows and WSL. A snapshot copies the SQLite database consistently.

## Self-development, personality, and limits

The forecaster predicts the change in open/detour success rates and normalized
mean return for an update block. Its forecast is recorded before training; its
error is measured before fitting that outcome. It is compared with predicting
no change at all. A few examples do not qualify it as a reliable developmental
planner, so forecasts remain advisory in this release.

Identity is a persistent name/UUID and provenance record. Individuality can be
studied through different experience histories, memories, and learned behavior.
The reported behavioral tendencies summarize observed collision/rest/resource/
discovery frequencies and also enter the recurrent input. They are not forced
personality adjectives, immutable trauma, or evidence of subjective experience.

The default runtime does not yet provide free-form language, learned temporal
options, reliable multi-step planning, autonomous code modification, or a proven
consciousness mechanism. The larger legacy architecture includes separate
language/option experiments; those do not silently become capabilities of the
new small runtime.

Relevant primary precedents: [DAgger](https://proceedings.mlr.press/v15/ross11a.html)
for corrective data collection, [homeostatic reinforcement learning](https://elifesciences.org/articles/04811)
for state-dependent resource regulation, and [model-based curriculum learning](https://proceedings.mlr.press/v232/xu23a.html)
for predicting the effects of training. This particular build is an engineering
research proposal, not a claim that its ingredients or combination are
unprecedented.
