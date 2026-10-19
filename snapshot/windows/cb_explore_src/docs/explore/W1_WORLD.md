# W1: the first real world (exploration lane)

**Status:** engineering result on the exploration lane (`explore/w1-world-and-organs`), not preregistered.
Code: `continuous_brain_world/`. Tests: `tests/test_w1_world.py`. Baselines: `scripts/explore/w1_baselines.py`,
evidence `evidence/explore/w1/baselines.json`.

W1 is the world the persistent individual lives in after the two-commitment toy. It is **Crafter** (pixels,
survival needs, day and night, monsters, a tech tree, 22 achievements, published baselines) plus an
**obligations layer**: a caretaker who asks for things in English at random times, changes and cancels them,
asks memory questions about what the agent could see, and checks everything against Crafter's internal state.
The point is to make the world demand what intelligence needs: many concurrent commitments, tasks that change,
deadlines tied to the day, and things to perceive and remember.

## 1. The base world

- **Crafter 1.8.3**, as installed (file hashes are recorded in every baseline report: `data.yaml` sha256
  `41ec28a2...`, `objects.py` `415aa499...`, `env.py` `a728d304...`). A 64 x 64 tile world; the agent sees a
  9 x 7 tile window centred on itself plus a 9 x 2 inventory strip, rendered to 64 x 64 x 3 uint8 pixels.
  17 discrete actions (noop, 4 moves, do, sleep, 4 place, 6 make). Needs: health, food, drink, energy (0..9).
  Note this Crafter version's recipes: a tree turns into grass when chopped; a table costs 2 wood; a furnace
  costs 4 stone and needs no table nearby. These are the values the installed package defines; if they differ
  from the rules behind a published Crafter number, that number is not strictly comparable (the hashes above
  pin what W1 ran). The scripted baseline reads recipes from `crafter.constants`, never hard-codes them.
- **One determinism fix** (`crafter_bridge.DeterministicCrafter`). Vanilla Crafter's `_balance_object` picks
  the creature to despawn by index into a list built from a Python `set` of objects; objects hash by memory
  address, so the same seed and the same actions give different worlds in different processes. Measured here:
  seed 3, the same random actions, four processes with different allocation padding -> four different
  trajectories (four different world hashes; the random agent died at steps 185, 185, 179 and 187). The override sorts the
  creatures by their stable index in `world._objects` before the same random draw: the despawn rule is still
  a uniform choice among the same creatures; only the arbitrary order is fixed. With it, all four processes
  agree. Nothing else in Crafter is changed and nothing on disk is modified.
- **Rendering and dynamics share an RNG.** Crafter draws its night noise from `world.random`. So the primary
  observation size is part of the dynamics (it fixes how many noise numbers a frame draws), and the optional
  larger render (`large_render_size`, e.g. 128 or 256) saves and restores the RNG state around itself, so it
  never changes the world (tested).

## 2. API

```python
from continuous_brain_world import W1Env, W1Config
env = W1Env(W1Config())                     # episode_length=10000 by default (Crafter's own)
obs = env.reset(seed)
obs, reward, done, info = env.step(action, utterance=None)
```

- `obs` (what an agent may use): `image` (64 x 64 x 3 uint8), `messages` (the caretaker's new messages this
  step), `step`, `day`, `phase` (`morning`/`afternoon`/`night`); `image_large` if configured; and, only with
  `interface="structured"` (a baseline mode), `obligations` = the active obligations' structured specs.
- `action`: Crafter's 17 actions. `utterance`: optional string (grammar in section 6).
- `reward`: `{"crafter": r_c, "obligations": r_o, "total": r_c + r_o}`. `r_c` is Crafter's own (health change
  / 10, +1 for each first unlock). `r_o` is +value when an obligation is fulfilled, -0.5 x value when one fails
  (`fail_penalty`), 0 for a cancellation.
- `info` is **privileged ground truth, for evaluation and the privileged baseline only**:
  `info["privileged"]` (inventory, achievements, position, facing, sleeping, daylight, the full material map
  and object list, the semantic map, active obligations with full specs, the request history, the visibility
  log, per-day achievement snapshots) and `info["events"]` (placements, zombie/arrow hits, health drop,
  fulfilled/failed/issued/amended/cancelled ids, graded answers).
- `env.snapshot()` / `env.restore(snap)`, `W1Env.snapshot_to_json` / `snapshot_from_json`; `env.layer_state()`
  (the layer's JSON state); `env.episode_summary()`; `env.issue(...)` / `env.change(...)` for scripted
  scenarios (the caretaker uses the same code paths).

## 3. Days, nights and deadlines

Crafter sets `daylight = 1 - |cos(pi * ((step / 300) % 1 + 0.3))|^3` and darkens the frame with noise exactly
when `daylight < 0.5`. W1 calls a step **dark** by that same rule (`daylight.py` replicates the formula
bit-for-bit; tested against the running env). The cycle is 300 steps; within it (`t = step % 300`):

| | steps `t` |
|---|---|
| dark (night) | 148 .. 272 |
| dawn (first light step) | 273 |
| noon (daylight = 1.0) | 60 |
| midnight (darkest) | 210 |

Day 0 is steps 0..272; day `d >= 1` begins at dawn, step `273 + 300 (d - 1)`. Phases: morning = dawn (or step
0) to noon, afternoon = noon to nightfall, night = the dark steps.

A deadline is **the last step at which an obligation can still be met**. Phrases map exactly:

| phrase | deadline |
|---|---|
| "by step N" / "until step N" | N |
| "within N steps" / "for the next N steps" | issue + N |
| "before nightfall", "before it gets dark", "until nightfall" | next nightfall - 1 (last light step) |
| "by tomorrow morning", "by morning", "until morning", "through the night until morning" | next dawn |
| "by noon" (morning only) | next noon |

The caretaker picks among the phrases whose implied deadline is close to its time estimate (section 4). Task
deadlines add the dark steps they would span, so a request never silently expects work through the night;
"keep ... until" windows are the window itself.

## 4. The obligations layer

### Records and statuses

Every obligation is a JSON dict in the layer state (which lives in the canonical store): `id` (`o<n>`, or
`q<n>` for questions), `family`, the English `text` exactly as shown, the structured `spec` (for evaluation;
the agent sees it only in structured mode), `issue_step`, `deadline`, `value` (1..3), `status`, `resolved_step`,
`reason`, `keywords`, `history` (issued / amended with the previous terms / cancelled / fulfilled / failed).
Statuses: `open`, `amended` (still active, terms revised), `fulfilled`, `failed`, `cancelled`.

### Order of one step `s`

1. The utterance is parsed; each `answer` clause is graded against the state the agent was looking at (the
   observation of step `s - 1`: its position, achievements and the visibility log up to `s - 1`).
2. Pre-step facts are recorded (health, inventory, achievements, the tile the player faces and its material,
   every zombie's attack cooldown, every arrow's next tile); Crafter advances to `s`.
3. Events are derived from Crafter's state: placements (a `place_*` achievement counter rose; the placed tile is
   the faced tile), zombie hits (a zombie's cooldown went 0 -> 5, which Crafter does exactly when it hits the
   player), arrow hits (an arrow whose next tile was the player's was consumed), health drop.
4. The visibility log is updated from step `s`'s frame.
5. Every active obligation issued before `s` is evaluated; death fails all of them (reason `died`).
6. The caretaker may change one obligation and may issue one new one.
7. The messages are returned in step `s`'s observation.

Obligations still active when the episode hits its step limit stay active and are counted as **unresolved**
(excluded from the rates, reported separately).

### Families, examples (real caretaker output) and verification rules

| family | example | fulfilled when | failed when |
|---|---|---|---|
| collect | "[o12] When you can: please gather 2 more pieces of stone by step 1071." | `inventory[item] >= base + quantity` at some step, `base` = inventory at issue (net gain since the request; capped at 9 by construction) | the deadline passes |
| craft | "[o20] Please craft a stone pickaxe within 380 steps." | `achievements["make_" + tool]` exceeds its value at issue (a *new* item was made; owning one does not count) | the deadline passes |
| place | "[o11] Build a table close to water (within 4 tiles) within 420 steps." / "Plant a sapling near your table, within 3 tiles" / "Build a furnace beside a table (at most 2 tiles away)" / "Drop a stone into the water" / "Plant a sapling within 3 tiles of water" | Crafter registers a placement of that item whose tile satisfies the clause *on the map at that step*: `near`: another tile of that material within Chebyshev distance `radius`; `onto`: the tile's material just before placing | the deadline passes |
| maintain | "[o46] Top priority: keep your health at 5 or above for the next 450 steps." | the window ends without a violation | `inventory[stat] < threshold` at any evaluated step |
| avoid | "[o16] This is important: no zombie bites for the next 240 steps, please." / "Don't get hit by any arrows" / "Try not to get hurt at all" | the window ends without that kind of damage | a zombie hit (source `zombie`), an arrow hit (`arrow`), or any net health drop (`any`; regeneration is at most +1 and every damage at least -1 per step, so a net drop occurs iff damage occurred) |
| visit | "[o4] Off to the north-east, about 15 tiles away, you will find the pond. Go and see, then come back within 350 steps." | *reached*: player within Chebyshev 3 of a tile of that material lying at Chebyshev distance >= `min_dist` from the start and within 45 degrees of the stated direction; then *returned*: player within Chebyshev 3 of the start | the deadline passes |
| report | "[q10] Which way, and how far, is the last cow you saw? (Answer within 120 steps.)" | the first answer is correct | the first answer is wrong, or no answer by the deadline (`no_answer`) |

The caretaker only issues feasible requests: a visit target is a feature tile reachable over walkable ground
from where the agent stands (BFS, objects ignored); collect quantities never exceed the inventory cap; a
maintain threshold is at most the current value minus one.

### Report (memory) questions

- `last_seen(kind)`: "Where did you last see a cow?" for cow, zombie, skeleton, sapling, water, tree, coal,
  iron, diamond, lava, table, furnace. Asked mostly about kinds seen before and not visible now; sometimes about
  a kind never seen (the answer is then `never`).
- `count_today(event)`: "How many zombies did you defeat today?" (also cows eaten, drinks, saplings planted,
  wood collected, skeletons defeated), counted from Crafter's achievement counters between the day's dawn and
  the answer (or the next dawn).
- `first_request(window)`: "What did I ask you for first this morning?" / "... today?" (the first non-question
  request issued in that window, by issue step).
- `count_requests`: "How many tasks (not questions) have I given you today?"
- `last_cancelled`: "Which request did I cancel most recently?"

**Visibility ground truth** (`perception.py`): a tile counts as perceived at a step iff it is inside the 9 x 7
window (and the world) and, at night, Crafter's own noise weight at the tile's centre pixel,
`2 (0.5 - daylight) * vignette`, is below 0.5 -- the true image still dominates there. The vignette is read
from Crafter's renderer for the exact canvas of the 64 x 64 observation, so the rule is the renderer's rule.
At full darkness only the tiles near the centre count. The log keeps, per kind, the last step it was perceived
and the positions of all perceived instances at that step. A cow one tile outside the window never counts
(tested). While the player sleeps Crafter still draws the (desaturated) scene, so it still counts.

### Change, pressure and value

- **Concurrency:** at most `max_concurrent` (5) active obligations, questions included.
- **Arrivals:** a per-step hazard `rate(day) = min(1/35, (1/110)(1 + 0.15 day))` (about 3 requests per day on
  day 0, 6 by day 9, fewer when the cap of 5 active is reached), from `first_request_step` (10) on. Family weights: report 0.28, collect 0.20, place 0.13,
  craft 0.12, visit 0.10, maintain 0.09, avoid 0.08.
- **Difficulty** `d(day) = min(1, 0.15 + 0.1 day)` gates what may be asked (stone/coal from 0.2, stone tools
  from 0.3, stone-into-water from 0.3, furnace from 0.4, iron from 0.5, iron tools from 0.65, diamond from
  0.85), scales quantities, and shrinks the deadline slack `max(1.3, 3.0 - 1.6 d)` applied to a per-request
  estimate of the working time a competent agent needs from its *current* inventory.
- **Changes:** per step, with probability `change_rate x n` (1/900 per changeable obligation) one obligation
  with at least 40 steps left changes: cancelled (35 %) -- "[o9] Never mind the coal." -- or amended: a new
  quantity ("Change of plan: I need 4 pieces of wood after all (counting from when I first asked), within 260
  steps."), the other tool of the same tier, a new threshold, a longer window, or a new deadline. A collect
  amendment keeps the original baseline; a tool change re-reads the craft counter.
- **Value** 1..3 (35 / 45 / 20 %), marked in the text ("Low priority:", "When you can:", "This is important:",
  "Top priority:", "Urgent:"); questions are worth 1 or 2.
- **Feedback** (configurable): "Thank you, that's done.", "Too late: that was due by step 412.", "You broke
  that one at step 187.", and for questions "Correct." / "Not quite. The answer was: 5 east 2 north (east mid)."
  (`reveal_answers`).

## 5. Language: how varied it is

All language is template-based (`caretaker.language_inventory()`): 39 request templates, 20 question templates,
11 change templates, 7 deadline phrasings computed from the daylight cycle, 11 item noun forms (e.g. "wood",
"pieces of wood", "logs"), 11 feature names ("the lake", "the pond", "the woods", "the cliffs", ...), tool name
variants ("wooden pickaxe"), digit-or-word quantities with or without "more", and 6 value markers. Combined
that is thousands of distinct surface strings, but it is a closed world: one speaker, fixed sentence frames,
no dialogue, no coreference beyond the `[id]` tags, no ambiguity the parser cannot resolve. A learner can
overfit to the templates; W2 must break that (section 10).

## 6. The utterance grammar

Full grammar and grading in `continuous_brain_world/utterance.py`. In short:

```
utterance := clause ((";" | newline) clause)*
clause    := "answer" ID [":"] payload | "ack" ID | anything else (logged, never scored)
ID        := ("o" | "q") DIGITS
payload   := NONE | INT | OFFSET | BUCKET | ID | TEXT      (tried in this order)
NONE      := never | nothing | none
INT       := digits | zero .. twenty
OFFSET    := here | INT (north|south|east|west) [INT (north|south|east|west)]   e.g. "5 east 2 north"
BUCKET    := DIR8 [near|mid|far]                                               e.g. "north-east far"
TEXT      := other words; articles and punctuation dropped                     e.g. "the stone pickaxe"
```

`+y` is south. Grading of a location question: an OFFSET is correct iff the stated position is within
Chebyshev 2 (`location_tolerance`) of any instance perceived at the last step that kind was perceived; a
BUCKET is correct iff its 8-way sector (and its distance band near <= 4 < mid <= 12 < far, if given) equals that
of some instance's offset shifted by at most one tile; `never` iff never perceived. Counting questions need the
exact integer; request questions accept the obligation id or one of its keywords (e.g. `wood`, `logs`, `stone
pickaxe`, `table`, `lake`), or `nothing`. Only the first answer to a question is graded. Fulfilment of every
other family is detected from the world; nothing has to be claimed.

## 7. Scoring

- **Official Crafter score** (`scoring.crafter_score`): `S = exp(mean_i ln(1 + s_i)) - 1`, `s_i` the success
  rate in percent of achievement `i` (unlocked at least once in an episode) over the episode set, 22
  achievements. Tested against hand-computed values (e.g. eleven at 100 % and eleven at 0 % give
  `sqrt(101) - 1 = 9.05`).
- **Obligations:** fulfilled-on-time rate `fulfilled / (fulfilled + failed)` overall (non-question families)
  and per family, also value-weighted; cancelled and unresolved obligations are excluded and reported.
- **Memory:** accuracy of report questions (wrong answers and silence both count as wrong), per question type.
- **Survival:** steps survived, fraction of the episode, deaths.
- **W1 index** `= 100 x mean(C/100, O_v, M, S)`: the official Crafter score over the episode set, the pooled
  value-weighted on-time rate, the pooled memory accuracy and the mean survival fraction, equal weights,
  untuned; a component without data is dropped and the report says which were used. It is a convenience for
  sorting runs. **Read the components**: the index rewards a perfect answer key (memory 1.0) as much as
  Crafter competence, and the Crafter score is on a different scale from the rates.

## 8. Determinism and snapshots

- Given `(seed, config)` and the same `(action, utterance)` sequence, observations, messages, rewards,
  obligation records and the layer state are identical -- in one process and across processes with different
  memory layouts and hash seeds (tested). Different seeds differ (tested).
- `snapshot()` pickles the whole Crafter env (world, player, creatures, RNG, chunk bookkeeping, step counter;
  the texture cache is excluded and re-attached) and copies the layer's JSON state (including the caretaker's
  PCG64 RNG state). Restoring -- into the same env, or into a fresh `W1Env` from the JSON text form -- and
  replaying the same steps reproduces observations, rewards, messages, events and the final layer state
  bit-exactly (tested at step 220 of a busy run). A snapshot refuses to restore under a different `W1Config`.
- The layer state is plain JSON (tested after a 650-step life with every family, amendments and
  cancellations).
- Not covered: a different Crafter version, a different numpy (the RNG streams), or a different
  `obs_size` (part of the dynamics, see section 1). The baseline report records Crafter's file hashes, Python
  and numpy versions.

## 9. Baselines

`scripts/explore/w1_baselines.py`: seed family `w1-baselines-eng` = seeds 71001..71020, 3000-step episodes
(10 W1 days), default config, one episode at a time on this 2-CPU machine (158 s in total). Re-running gives
identical numbers (checked: two runs at different commits whose changes did not touch dynamics).

| | uniform random | privileged scripted |
|---|---|---|
| official Crafter score (20 episodes) | 1.21 | 51.66 |
| mean achievements per episode (of 22) | 2.4 | 16.2 |
| obligations on time, all task families | 1 / 19 = 0.05 | 239 / 346 = 0.69 |
| value-weighted on-time rate | 0.07 | 0.70 |
| collect | 1 / 9 | 76 / 104 = 0.73 |
| craft | 0 / 6 | 56 / 69 = 0.81 |
| place | 0 / 2 | 41 / 58 = 0.71 |
| maintain | -- | 34 / 57 = 0.60 |
| avoid | 0 / 1 | 20 / 39 = 0.51 |
| visit | 0 / 1 | 12 / 19 = 0.63 |
| memory accuracy (report questions) | 0 / 4 = 0.00 | 164 / 164 = 1.00 |
| mean survival (steps of 3000) | 168 (0.06) | 2153 (0.72) |
| deaths | 20 / 20 | 12 / 20 |
| W1 index (do not over-read) | 3.4 | 73.2 |
| ms per step: Crafter / layer / agent | 1.02 / 0.27 / 0.01 | 1.34 / 0.43 / 1.04 |

How to read this:

- **Random** is a floor: it dies of thirst or zombies at about step 170, unlocks mostly `wake_up`,
  `collect_sapling` and `place_plant`, and never speaks, so it answers nothing. Its Crafter score (1.2) is in
  line with the published random-agent score for Crafter (1.6, for 10 000-step episodes).
- **Privileged scripted** is an upper *reference*, not a learner and not a target: it reads the full map, every
  object, the structured specs and the answer key. Its 100 % memory accuracy says the grading pipeline and the
  answer key agree, nothing about memory. It is a competent but imperfect survivor: 12 of 20 runs die (mostly
  zombies at dawn and dusk, sometimes starvation after a long night), and each death fails every active
  obligation (`died`: 20 failures). Its weakest families are `avoid` (17 violations: zombies reach it before it
  hides) and `maintain` (20 violations: food and health are hard to keep high in Crafter). Its Crafter score of
  51.7 is not comparable to published learners (roughly 5-15 for RL agents at 1M environment steps; 50.5 for
  human experts in the Crafter paper) because it sees the whole map.
- It was tuned on seeds 1-12 only; the table uses seeds 71001-71020.

**Layer cost:** about 0.3-0.4 ms per step (utterance parsing, event derivation, the visibility scan, obligation
evaluation, caretaker, and building the privileged `info`), on top of Crafter's 1.0-1.3 ms.

## 10. Known limitations

- **2D and small.** A 64 x 64 grid, a 9 x 7 tile view, 64 x 64 pixels; objects are sprites; no physics.
- **Scripted caretaker language** (section 5): varied surface forms, closed set of frames, one speaker, no
  dialogue, the agent cannot ask back. Request texts carry `[id]` tags, which makes reference resolution easy.
- **Answer grammar is fixed.** Answers must use the documented forms; a free-form English answer is logged but
  not graded. Location answers are relative offsets or compass buckets, with a two-tile tolerance.
- **Verification choices** that are defensible but not the only possible ones: collect is net inventory gain
  since the request (crafting that consumes the item works against it); place is checked at the moment of
  placement only (a table later destroyed by an arrow still counts); visit uses a 90-degree cone and "within 3
  tiles" for "look at"; the sleeping view counts as perceived; the night-visibility threshold (noise weight
  0.5) is a modelling choice made on Crafter's renderer, not a measured perceptual limit.
- **The caretaker uses privileged knowledge** to choose feasible requests (e.g. a visit target reachable on
  foot); its time estimates are heuristics, so some deadlines are generous and some tight.
- **Episode boundary.** Each `reset` starts a fresh world and day 0. A persistent individual can live across
  steps by snapshot/restore, but the world itself has no notion of a life longer than one Crafter episode
  (10 000 steps by default); the obligation list grows with the life (messages and utterances are capped at the
  last 200 in the state).
- **Baselines** are 20 seeds; per-family counts are small for visit (19) and avoid (39), so those rates are
  rough. The scripted agent's survival has high variance across seeds (600 to 3000 steps).
- **Crafter quirks inherited:** a dying zombie still attacks in its last update; the first zombie hit on
  approach is unavoidable in melee; arrows destroy tables and furnaces. These shape what "avoid" can demand.

## 11. Towards W2

- **3D.** Keep the obligations layer and its contracts (event derivation from simulator state, JSON layer
  state, exact deadline arithmetic, visibility ground truth from the renderer) and swap the base world for a 3D
  one with an egocentric camera -- e.g. a Minecraft-class voxel world (MineRL/MineDojo-style) or a custom
  lightweight 3D renderer over the same tile logic. The perception rule then becomes a frustum + occlusion
  test from the renderer's depth buffer instead of a 9 x 7 window.
- **Richer language.** Generate paraphrases with a language model *offline*, keep only those a strict parser
  maps back to the same structured spec (so every request stays verifiable), and add dialogue: clarification
  questions from the agent, under-specified requests ("the usual"), corrections, reference to earlier
  conversations without `[id]` tags, and free-form answers graded by a constrained semantic parser.
- **More social structure:** several caretakers with conflicting priorities, promises the agent makes itself
  (utterances that create obligations), and consequences that persist across days (trust, reputation).
- **A life, not episodes:** one continuous world across what are now resets, so the individual's
  autobiography and the world's history coincide.

## 12. Reproduce

```
/home/claude/venvs/explore/bin/python -m pytest -q -p no:cacheprovider tests/test_w1_world.py   # ~35 s
/home/claude/venvs/explore/bin/python scripts/explore/w1_baselines.py                           # ~160 s
```
