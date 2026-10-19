# Continuous Brain v0.10.1 Architecture

## Purpose

v0.10.1 keeps the v0.10.0 dedicated recurrent planning cortex and repairs only its action-value learning rule.

Development evidence from the already-used detour seeds showed that the v0.10.0 recurrent transition learned the action semantics correctly in all 10 tested seeds at both the initial state and immediately before the wall: `1=up`, `2=down`, `4=right`. Yet the learned value/Bellman/goal signals ranked **UP** best 0/10 times, and the old Monte-Carlo Q head ranked UP best only 1/10 times. The planning dynamics were therefore retained rather than redesigned.

## What was wrong with v0.10.0 Q supervision

v0.10.0 trained:

```
Q(s, executed_action) ~= discounted return produced by the nursery collector afterward
```

The detour collector deliberately explored and often recovered after poor actions. A bad action could therefore receive a fairly good target because later behavior corrected the mistake. That target estimates the collector's behavior return, not the best future achievable after the action.

## v0.10.1 Bellman Q

v0.10.1 trains only `planning.q_head` with fitted offline Bellman targets from actual lived transitions:

```
target = reward + gamma * max_a Q_target(actual_next_state, a)
```

Terminal transitions omit the bootstrap term.

The target Q network is a slowly moving copy of the child's Q head. It is training scaffolding only and is not a separate runtime cognitive organ.

## Conservative offline safeguards

Three safeguards are fixed into the training algorithm:

1. **Balanced action sampling** — each minibatch samples actions approximately uniformly so common collector actions cannot dominate rare-but-supported alternatives.
2. **Conservative penalty** — `logsumexp(Q(s,*)) - Q(s,a_lived)` penalizes unsupported optimistic counterfactual values.
3. **Lived-return support clamp** — Bellman targets are clamped to the minimum and maximum discounted returns actually observed in the training nursery. The offline learner cannot invent values outside demonstrated return support.

A small consistency loss also makes Q agree between the actual encoded next latent and the v0.10 recurrent planner's own predicted one-step latent. This is not dynamics training; all recurrent planning tensors remain frozen.

## Migration and identity continuity

No parameter shape changes between v0.10.0 and v0.10.1. There is no new child organ. A v0.10.0 checkpoint is copied directly into a v0.10.1 state directory.

During `train-planning-q`:

- all pre-v0.10 child tensors are frozen;
- all v0.10 recurrent planning tensors are frozen;
- only `planning.q_head.*` is trainable;
- the trainer verifies after optimization that every non-Q tensor is bit-for-bit unchanged and aborts on any drift.

## Runtime planning

The existing v0.10 recurrent beam planner is retained. Its terminal/intermediate heuristic now consumes the conservative Bellman Q head rather than the old behavior-return Q head.

No parents are used during planning qualification. Evaluation performs no training and writes no autobiographical memory.

## Scientific status

P3 predictive world-model qualification remains passed from the frozen lineage. P4a consequence-aware behavior remains passed. P4b successful detour completion remains open until an untouched blind test satisfies the predeclared gate:

- absolute completion >= 20%;
- completion advantage over reactive > 5 percentage points;
- collision regression <= 2 per episode.

No claim of consciousness, sentience, personhood, or AGI is made.
