# The individual in W1 (exploration lane, ENGINEERING_ONLY)

The first time the Continuous Brain individual lives in a world with real demands. W1 (`docs/explore/W1_WORLD.md`)
is a deterministic Crafter world with survival, a tech tree, day and night, monsters, and a caretaker who sends
requests in English, changes them, sets deadlines and asks questions about the past. Nothing here is a scientific
result: the exploration lane is engineering, and every run is labelled so. The protected I1 lane is untouched
(`LANES.md`).

## 1. What the individual is

One `UnifiedRuntime` (the same runtime and store-authority invariant as I1). Each faculty is a host whose persistent
state lives in the canonical store as the blob channel `<slot>.state`. It is read at the start of each tick and
written back at the end; nothing about the individual is kept in Python between ticks.

| slot | faculty | what it does | state in the store |
|---|---|---|---|
| perception | **vision organ** | 64×64 pixels → the perceived 9×7 tile map (materials and creatures) | none (pinned weights) |
| episodic_memory | **memory** | integrates confident perception into a remembered world map; last-seen log per kind; remembered cows and plants; its own achievement counts at the start of each day | map, last-seen, objects, day snapshots |
| goal_genesis | **commitment ledger** | the language organ turns each caretaker message into events; the ledger keeps what it was asked, amendments, cancellations, questions, feedback, and what it did not understand | commitments, questions, request log |
| self_state | **self-model** | believed success rate per kind of commitment (from the caretaker's feedback), and vitals | success counts |
| metacognition | **speech** | answers questions from its *own* memory (never from the world's records), acknowledges requests | what it has answered |
| skills | **skills** | chooses each action from its own beliefs (below) | routine state, exploration target, visit progress, pain memory |
| autobiography | **autobiography** | logs each day's events and narrates the day at dawn (language organ); expected vs actual fulfilment | days narrated |

**Body sense is the only privileged read:** vitals, inventory, position, facing, sleeping and its own
achievement counters. `GuardedInfo` raises on anything else. A test runs a life in which every other privileged
field is replaced by garbage, and the life is identical.

### What is its own, and what is borrowed (be exact about this)

| faculty | status in v1 |
|---|---|
| perception | **learned** from pixels (a supervised developmental phase with teacher labels; `VISION_ORGAN.md`) |
| memory, ledger, self-model, speech, autobiography | **its own mechanisms**, built from what it perceives and hears |
| language understanding | **an organ**: on this CPU, the development stand-in that knows W1's templates (labelled everywhere); on a machine with weights, an open-weight model (`LANGUAGE_ORGAN.md`) |
| skills | **borrowed**: W1's privileged reference routine library (survival, shelter, gathering, tech tree, commitments), run on the individual's *beliefs* instead of the true world. Added on top: frontier exploration, a provisioning search when food or water is unknown, avoiding tunnel frontiers, eating food in sight, a pain response (leave a shelter where it keeps being hurt for no reason it can feel; remember the spot; distrust water walls for a while) |

The skills are the obvious next thing to learn (§5).

To watch one life as it happens (what it sees, remembers and intends, and each faculty's activity in the store),
use the live viewer: `scripts/explore/w1_viewer.py` (`docs/explore/README.md`, "Watching it live").

## 2. Invariants (tests: `tests/test_w1_individual.py`)

- **No hidden state.** Rebuild the whole individual (new hosts and organ objects) from its saved state every
  25 steps: the life is identical to the uninterrupted one.
- **Persistence.** Save the individual and the world mid-life, restore both into fresh objects, and the next 70
  steps and the final state hash are identical.
- **Store authority.** An ERASE of `episodic_memory.state` at tick 100 leaves the store with no memory. At
  the next tick the individual remembers only its current view, and its life diverges from its untouched twin.
  An ERASE of `goal_genesis.state` makes it forget every commitment it had made.
- **The body boundary.** As above: a poisoned world, the same life.
- **Determinism.** The same seed and arm give the same life (the experiment script reproduces its own numbers).

## 3. Results (20 worlds, seeds 71001–71020, up to 3000 steps each)

Run: `python scripts/explore/w1_individual.py --seeds 71001-71020` at commit `a95fb9c`, clean tree, vision weights
`0c46b2d1…`, language = the development stand-in. Evidence: `evidence/explore/individual/w1_individual_20.json`.

| arm | Crafter score | achievements / life | requests on time | memory answers | mean survival (steps) | deaths |
|---|---|---|---|---|---|---|
| random | 1.2 | 2.4 | 1/19 (0.05) | 0/4 | 168 | 20 |
| privileged reference (sees the true world) | 51.7 | 16.1 | 239/346 (0.69) | 164/164 | 2153 | 12 |
| **the individual** | **30.2** | **12.8** | **56/98 (0.57)** | **29/29** | **688** | 20 |
| individual, perception = oracle | 30.1 | 12.8 | 51/89 (0.57) | 28/28 | 656 | 20 |
| individual, no memory | 16.6 | 9.6 | 20/40 (0.50) | 4/8 | 307 | 20 |
| individual, no language | 25.3 | 12.8 | 25/103 (0.24) | 0/42 | 735 | 20 |

By family, the individual against the reference (fulfilled / failed): collect 18/7 vs 76/28; craft 16/8 vs 56/13;
place 9/7 vs 41/17; maintain 9/6 vs 34/23; avoid 2/9 vs 20/19; visit 2/5 vs 12/7. Every memory question it was
asked while alive it answered correctly from its own memory, graded by W1 against what it had actually seen.

A day from one of its lives, narrated by itself at dawn (seed 71003, development narrator):

> At step 4 (morning of day 0) I managed to collect wood for the first time. … At step 119 (afternoon of day 0) I
> was asked to gather 2 wood by step 439 [o1]. … At step 248 (night of day 0) I was asked to keep my food at 2 or
> above until step 518 [o3].
>
> At step 315 (morning of day 1) I was told that I had broken [o3]. At step 358 (morning of day 1) I was told that
> [o1] was done. …

Its self-model at the end of that life: avoid 0 of 2, collect 2 of 3, maintain 1 of 2, memory questions 6 of 6.

Readings:

- **It is a real middle ground.** It perceives from pixels, remembers, understands requests and answers from memory.
  Its score is 25× random and 58% of the privileged reference, which sees the true world.
- **The vision organ is not the bottleneck.** With the true tiles of its view, nothing changes (30.1 vs 30.2).
- **Memory is worth most of what it does.** Without it the score nearly halves (16.6), survival more than halves
  (307 steps), and it can answer only questions about what is in view at that moment (4 of 8).
- **Language is what makes it keep its word.** Without the language organ its on-time rate falls from 0.57 to 0.24
  (the rest is accidental), and it can answer none of 42 questions. Its Crafter score drops too (25.3), because
  requests also pull it along the tech tree.
- **What it lacks is foresight under partial observability.** It dies in every world, at 688 steps on average
  against 2153 for the reference. The causes, traced life by life: starving or going thirsty when it knows no food
  or water nearby, and zombies and skeleton arrows at night or near tunnels. The reference avoids these by seeing
  every cow and skeleton on the map. Closing that gap is a learning problem: knowing in advance where food will be,
  and when to provision before the night (§5).
- **The numbers come from the template stand-in**, not a language model. With a real open-weight model they can
  only stay the same or get worse at first. That is the next measurement.

## 4. What did not work (kept, so nobody repeats it)

- **Acting on bump memory.** The memory notices when a move does not move it: the tile ahead is occupied. I first
  treated those tiles as walls for 60 steps, then as short-lived obstacles for 20. Both made lives much shorter
  (10 worlds: Crafter score 7.2, then 18.2, against 32.1 without). Most bumps are animals that move on, and the
  routines then plan around phantom obstacles. Bumps are still recorded in the memory, but not acted on
  (`skills.USE_BUMPS = False`).
- **Writing unsure perceptions into the map** (a best guess for a never-seen tile) made no difference on 10 worlds;
  it is off (`memory.GUESS_UNSURE = False`).
- An early action-index error (move actions are 1–4 in Crafter) briefly inverted the bump logic; caught by the
  numbers collapsing and fixed with the real action table.

## 5. Next

1. **A real language model.** Run the same experiment with `--language hf` on David's WSL (CPU is enough for
   Qwen2.5-0.5B: about 30 messages per life) or a rented GPU; then the larger models of the ladder. This
   replaces the stand-in and measures open-model understanding inside a living individual.
2. **Learn the skills.** First imitation of the privileged reference from the individual's own beliefs (DAgger:
   the teacher labels the states the student visits), then reinforcement learning on the full W1 reward, with
   a learned world model for foresight (provisioning before the night is the biggest gap). This needs a GPU.
3. **Perception beyond tiles.** An open-weight vision-language model (SmolVLM, Qwen2-VL) to describe what it sees
   in words, feeding the autobiography and memory answers.
4. **Private thoughts.** A sealed, committed channel in the store for its own deliberation (the language organ
   thinking to itself), readable by David when he chooses, never by the evaluator.
5. **W2.** A richer world (3D, more agents, open language) once the W1 individual survives and learns.
