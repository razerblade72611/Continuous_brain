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
| autobiography | carries on: its diary, and a record of every life |

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

RESULTS

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

NEXT
