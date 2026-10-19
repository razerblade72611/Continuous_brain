# One individual, many lives: learning from its own experience (exploration lane, ENGINEERING_ONLY)

The W1 individual (`INDIVIDUAL_W1.md`) lived one life with borrowed skills and learned nothing from it. Here it
becomes one individual across lives, and two of its faculties learn, only from what happens to it. Nothing here is a
scientific result. The protected I1 lane is untouched (`LANES.md`).

## 1. One individual across lives

When it dies, its last moment is one more tick, in which its body sense tells it that it is dying. Its faculties
close the life:
- its last decision's return ends in death;
- what it was waiting to learn about the future is resolved;
- it consolidates what it learned;
- its autobiography records the life ("my life ended with no water left").

Then it wakes in a new world as the same runtime and the same store. Each faculty decides what belongs to the old
world:

| faculty | in the next life |
|---|---|
| memory | starts from nothing: nothing it saw holds in the new world |
| commitments, speech | start over: a new caretaker, new request ids |
| skills (routine plans) | its shelter and plans are gone |
| skills (what it learned), foresight | carry on |
| self-model | carries on: what it believes about itself |
| autobiography | carries on: its diary (the most recent 120 days) and a record of its last 300 lives |

The driver tells it two things about its situation: which life this is, and in its last moment, that it is dying.
These are labelled as the world's news, like its clock.

## 2. What learns, and from what

Both learning faculties see a fixed vector of 62 features built only from its beliefs (`features.py`):
- its vitals and the time of day;
- how near remembered water, trees, stone, coal, iron, its table and furnace are;
- cows, plants and monsters in view;
- tunnels and places that hurt it before;
- its tools and stocks;
- how much of the world it knows;
- recent damage and trends;
- its open commitments and deadline pressure.

Nothing privileged, no teacher.

**Foresight** (`world_model`). Every 5 steps it notes these features. 100 steps later (or when it dies) it learns what
followed: did it die, run out of water, run out of food, lose 3 or more health, run out of energy. A small network
(62→48→48→5, logistic loss) predicts those risks now, and its predictions are part of what its choice sees. Its
error is tracked against each outcome when it arrives.

**Choice** (`skills`). Its skills are courses of action (options) built from the routine library:
- its routine;
- drink;
- eat;
- rest in a shelter;
- get away from threats;
- explore;
- its requests;
- build and make things.

An option runs until it is done, cannot go on, or something salient happens: it is hurt, water or food falls to 3,
night falls, or the caretaker asks something new. Then it chooses again, by a Boltzmann distribution over its
learned option values plus a prior that favours its routine, mixed with a small, decaying chance of trying any
feasible option. So it starts where the routine is and learns when to depart from it.

What an option is worth is what it felt while the option ran:
- health gained or lost (0.1 a point);
- each new thing done (1);
- the caretaker's thanks or complaints (+1 / −1);
- discounted by 0.995 per step to the end of the life;
- dying ends that sum with −5.

When a life ends, every decision in it gets the return that followed it, and the value network (67→96→96→8: the 62 features and its 5 foresight predictions in, one value per option out) is fitted
to the returns of the decisions in its memory (Monte Carlo, Huber loss, Adam).

Some sense is innate. In the dark it does not wander off to explore, build or run errands, and it does not build a
shelter in daylight unless it is tired. Within that, which option serves it best is learned.

**Where it lives.** Weights, Adam moments and memories are float32/float16 vectors in the canonical store
(`world_model.weights`, `world_model.memory`, `skills.weights`, `skills.memory`). Bookkeeping is in the faculties'
`.state` blobs. Saving, restoring, erasing and hashing work on what it learned exactly as on everything else.

## 3. Invariants (tests: `tests/test_w1_learned.py`, `tests/test_w1_viewer.py`)

- **Exact machinery.** Backpropagation matches finite differences; Adam's first step and the losses are checked.
- **No hidden state while learning.** Rebuilding the whole individual from its saved state every 25 steps gives the
  identical life and final state hash.
- **A life ends, the same individual continues.** After its last moment and the next life's first ticks:
  - its memory of the old world and its commitments are gone;
  - its self-model and diary are the same;
  - what it learned crossed over unchanged.
- **Frozen means frozen.** With `learn=False` neither network changes over a life.
- **Erasing its skills state forgets what it learned.** Its weights are the newborn's again.
- **Its routine, always chosen, is the routine individual.** Forcing the routine option makes every action equal to
  the v1 individual's, over a whole life (and v1 itself reproduces its earlier lives bit for bit after the
  refactor).
- **Watching does not change it**, and a death in the viewer is saved and resumes exactly.

## 4. Results

Two measurements, both with the development language stand-in. Evidence:
`evidence/explore/lifelong/lifelong_summary.json`.

- **While it learns.** Training lives on worlds 73001 onward, each compared with the routine individual (`INDIVIDUAL_W1.md`) living the same world.
- **What it has learned.** Saved checkpoints tested frozen (no learning, no random tries) on the 20 test worlds of `INDIVIDUAL_W1.md` (71001–71020), which it never lived in.

**The shipped design** (bootstrap ensemble, confidence-gated; run8, 162 lives, about 80,000 steps). Training lives in
blocks of 25:

| lives | steps lived (it / routine, same worlds) | things done (it / routine) | requests kept (it / routine) | routine chosen |
|---|---|---|---|---|
| 0–24 | 448 / 662 | 13.4 / 12.4 | 18/66 / 55/102 | 59% |
| 25–49 | 521 / 590 | 13.7 / 13.6 | 36/82 / 54/103 | 71% |
| 50–74 | 420 / 535 | 13.6 / 12.4 | 31/67 / 38/84 | 68% |
| 75–99 | 660 / 697 | 13.6 / 13.0 | 67/120 / 67/132 | 70% |
| 100–124 | 489 / 559 | 13.0 / 12.7 | 40/83 / 51/95 | 77% |
| 125–149 | 479 / 643 | 13.1 / 12.5 | 37/60 / 49/99 | 78% |

Tested frozen on the 20 test worlds:

| | Crafter score | things done per life | requests on time | survival (steps) |
|---|---|---|---|---|
| routine individual | 30.2 | 12.8 | 0.57 | 688 |
| newborn (0 lives) | 30.2 | 12.8 | 0.57 | 688 |
| after 100 lives | 30.1 | 12.8 | 0.47 | 481 |
| after 150 lives | 32.2 | 13.2 | 0.49 | 605 |

**An earlier design** (one network, Boltzmann choice with a prior, a 200-step horizon, no well-being; run4), tested
the same way:
- newborn: 35.3 score / 762 steps;
- after 50 lives: 35.7 / 576;
- after 100 lives: 27.4 / 468.

Its newborn differs from the routine because its random values already pull it off the routine now and then.
That difference alone shows how much 20 worlds move with small changes of behaviour.

Readings:

- **The machinery learns, and what it learns is real.** Its foresight's error falls with experience. Its choice
  fits the returns that follow its decisions. It departs from its routine in situations where its memory says
  departing paid.
- **It does more, but it does not yet live longer than its routine.** Every training block shows more things done
  than its routine on the same worlds. Every block also shows shorter lives, and fewer of its requests kept. On the
  test worlds, after 150 lives, its Crafter score is 2 points above the routine and its lives 83 steps shorter.
  With 20 worlds, both differences are within what small changes of behaviour move.
- **Why survival lags.** Its returns look about 50 steps ahead (a discount of 0.98 per step). A longer horizon
  made the returns too noisy to learn from. At that horizon, a new thing done now outweighs a slightly higher
  chance of dying 100 steps later. Its deaths are mostly need management: thirst and hunger together cause about
  60% of them, the rest are monsters. The fix for those is partly below the options it chooses among. The routine
  waits in a besieged shelter until thirst kills it, and does not know where to find water. Choosing between the
  same routines more cleverly cannot fix that.
- **The newborn is its routine.** With no evidence it never leaves its routine, so a new individual starts exactly
  where the v1 individual was, and anything it learns is measured from there.

### 4b. David's overnight run: 1104 lives on his machine (2026-10-04/05)

One individual, the shipped design, in the viewer with "Next life when it dies". It lived 1104 lives (652,089
ticks) in worlds 71001–72104. Its first 20 lives were in the 20 test worlds above, so for this individual those
worlds are no longer held out. The fresh test below uses worlds 74001–74040, which no individual has lived in.
Evidence: `evidence/explore/lifelong/home_overnight/`.

While it learned, against the routine individual in the same worlds:

| lives | steps lived (it / routine) | difference (standard error) | things done (it / routine) | difference (s.e.) | requests kept (it / routine) |
|---|---|---|---|---|---|
| 0–99 | 485 / 617 | −132 (38) | 13.6 / 12.9 | +0.66 (0.25) | 55% / 50% |
| 1000–1099 | 608 / 662 | −54 (47) | 13.2 / 12.1 | +1.05 (0.28) | 50% / 46% |

Tested frozen on 40 fresh worlds (74001–74040):

| | Crafter score | things done per life | requests on time | survival (steps) |
|---|---|---|---|---|
| routine individual | 23.1 | 11.6 | 0.44 | 581 |
| after 1104 lives | 29.1 | 12.7 | 0.38 | 608 |

On the same 40 worlds, paired: survival +27 steps (standard error 51); things done +1.05 per life (standard error
0.29).

How it got there:
- It left its routine at about 7.5% of its decisions, mostly to build, to explore, to drink or to rest.
- In 15 of the 40 worlds it never left its routine, so those lives were identical to the routine's.
- Its deaths over the run: thirst 42%, monsters 38%, hunger 20%. Neither thirst nor hunger fell.

Readings:
- **It learned something real and modest.** It does about one more new thing per life than its routine, in
  worlds it has never seen; that difference is about 3.6 standard errors.
- **What its early exploring cost in survival, it learned back.** The gap was 132 steps in its first 100 lives and
  54 in lives 1000–1099. Frozen, on fresh worlds, it lives as long as its routine; it does not live longer.
- **It keeps somewhat fewer requests on time** (0.38 against 0.44).
- **Its need deaths did not fall.** As section 4 expected, choosing among the same routines does not fix thirst
  and hunger. The learned skills and the world model (section 6) are what should.

## 5. What did not work (kept, so nobody repeats it)

- **Bootstrapped option values.** Semi-Markov double Q-learning (a target network, bootstrapping at each decision,
  about 20 updates per stored decision) made its values run away. After 25 lives the average option value was 85
  and the largest 631. The real returns of an option averaged 0.3, and a whole life's about 10. Replaced by Monte
  Carlo returns over each life: lives here are short and always end, so nothing needs to be bootstrapped.
- **No forced exploration.** With only a temperature and a prior, an option it never tried kept its initial value,
  fell behind as the others learned, and was never tried. "Drink" was chosen 0 times in 8 lives. A small, decaying
  chance of trying any feasible option fixed it.
- **Every option at any time.** Early on, it explored and built in the dark and built shelters in daylight when not
  tired, and lived much shorter lives than its routine (423 steps against 643 on the same worlds). The innate
  limits above stopped that.
- **numpy's threads.** For these small networks, BLAS threads made each learning update about 6 times slower on 2
  CPUs. The scripts pin one thread.

## 6. Next

1. **Far more lives.** 160 lives here are about 80,000 steps. Deep reinforcement learning agents in Crafter are
   usually trained for a million. On David's machine, `scripts/explore/w1_lifelong.py train --lives 2000` runs
   overnight. Then `w1_lifelong.py evaluate --snapshot ...` measures it, and `w1_viewer.py --adopt ...` lets him
   watch the trained individual.
2. **A longer horizon that is still learnable.** Bootstrapped targets taken only from the ensemble's lower bound,
   and per-state baselines (its routine's estimated value), so that the next 300 steps count without the variance
   that sank the 0.995 horizon.
3. **Learning below the options.** Learn the skills themselves: imitation of its routine from its own beliefs, then
   reinforcement learning with a learned world model, on a GPU. This is what fixes the thirst deaths and the
   besieged shelters.
4. **A real language model in the loop** (`README.md`), now with an individual that keeps what it learns.
