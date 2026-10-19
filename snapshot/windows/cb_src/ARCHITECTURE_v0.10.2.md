# Continuous Brain v0.10.2 Architecture

## Purpose

v0.10.2 adds child-owned temporal abstraction after v0.10.1 demonstrated that the child had learned important pieces of the detour but could not organize them into a stable multi-phase plan.

Development evidence on already-used seeds showed:

- direct Bellman-Q reached the gap band in 7/10 episodes;
- direct Bellman-Q crossed the wall in 2/10 episodes;
- when left above the gap it continued choosing the same upward behavior rather than terminating the approach phase;
- a privileged crossing intervention increased crossings to 7/10 but still produced 0/10 completions because post-cross behavior did not reliably transition to final target approach.

The missing capability is therefore treated as **temporal organization / phase commitment**, not as a wall-physics or action-semantics problem.

## Option / Intention Cortex

The new `OptionCortex` is a persistent child-owned neural organ with four learned functions:

1. `selector` — chooses an anonymous latent option from the current planning state;
2. `termination` — predicts whether the currently active option should end;
3. `action_policy` — predicts primitive actions conditioned on state and active option;
4. `q_head` — estimates option-conditioned action value for imagination/search.

An option has a learned embedding and remains active across primitive timesteps until its termination signal fires after a minimum commitment period.

## No hand-coded detour phases

Training does **not** label states as `approach_gap`, `cross_wall`, `go_down`, `go_right`, or any equivalent semantic rule.

Successful lived trajectories are converted into short future-motion descriptors containing:

- change in structured physical state over a short future window;
- the action distribution observed during that future window;
- window duration;
- accumulated lived reward.

These descriptors are standardized and clustered with deterministic k-means. Cluster IDs are anonymous option identities. A small temporal smoothing step removes single-step cluster noise. The child then learns to infer those option identities from its own planning latent and to terminate/switch them from state.

The fixed wall x-coordinate and gap y-coordinate never enter option discovery or option training.

## Training isolation

`train-options` freezes every parameter outside `options.*`.

The trainer snapshots all pre-v0.10.2 tensors and aborts if any of them changes. Thus the following remain bit-for-bit unchanged during the v0.10.2 proof phase:

- vision and spatial cortices;
- language;
- nucleus/global workspace;
- motor cortex;
- world model;
- v0.10.0 recurrent planning dynamics;
- v0.10.1 conservative Bellman Q;
- values and prior developmental organs.

## Runtime temporal organization

At each lived state:

1. the child encodes the real dorsal/structured state into the v0.10 recurrent planning latent;
2. the option cortex selects an option if none is active, or evaluates termination of the active option;
3. the active option is held for at least `min_commit` primitive steps before termination can switch it;
4. option-guided planning recurrently imagines candidate primitive actions;
5. beam ranking uses imagined discounted rewards plus the terminal **option-conditioned action value**;
6. the first primitive action is executed in the real world;
7. the option identity persists into the next lived timestep unless its learned termination condition is met.

The release also evaluates `option_direct` as a diagnostic ablation. Formal P4b evidence comes from `option_planning`, which still uses the child-owned recurrent imagination system.

## Scientific interpretation

Because options are learned from successful lived action/state sequences, a v0.10.2 P4b pass would demonstrate learned temporal abstraction plus model-based option-guided planning in this detour regime. It would not demonstrate zero-shot planning from no prior successful experience, nor broad/general planning across arbitrary layouts.

The P4b gate remains unchanged:

- absolute completion >= 20%;
- completion advantage over reactive > 5 percentage points;
- collision regression <= 2 collisions per episode.

No claim of consciousness, sentience, personhood, or AGI is made.
