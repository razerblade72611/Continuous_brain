# CB2-S1: the scale-first kernel (exploration lane, ENGINEERING_ONLY)

CB2-S1 replaces the individual's hand-written mind with learned parts built on proven open-source work, and keeps
everything that makes it an individual. It follows the plan agreed on Linear, "Continuous Brain 2 — Scale-First
One-Stream Build Plan", and the One Stream design doc. Nothing here is a scientific result. The protected I1 lane
is untouched (`LANES.md`).

## 1. Four contracts

Everything behind these can be replaced without changing who the individual is.

| contract | module | what it fixes |
|---|---|---|
| the archive | `continuous_brain_cb2/archive.py` | Parquet shards with two-axis provenance, manifests and a lived hash chain |
| the core | `continuous_brain_cb2/world_model.py` | `initial`, `step`, `scan`, `imagine`, `feat` |
| the organ interface | the existing organ protocol, plus `CoreSkillHost` for the skill slot | how faculties read and write the store |
| consolidation | `continuous_brain_cb2/consolidation.py` | the one recorded event that changes what it has learned |

## 2. The archive

One directory per stream. `where` is per stream: LIVED (its one chronology), PRACTICE (a real simulator, not
lived), DREAMED (imagined). `who` is per row: SELF or TEACHER. A COUNTERFACTUAL label is the teacher namespace's
`teacher_action` when the teacher did not act.

- **Namespaces:**
  - `student`: frame, body sense, messages, action, utterance, reward, ending, the store hash after the tick;
  - `teacher`: the routine's action and option, and the true tiles of the view;
  - `organ`: its own memory read out as 62 belief features (no routine state) and a 33 x 33 window of its
    remembered map.

  Readers name the namespaces they open.
- **Shards:** 65,536 transitions, row groups of 4,096, zstd. Measured on W1: 2.2 KB a transition, read at 21,000 a
  second per core. A lived life ends a shard, so a crash never loses a sealed life.
- **Enforced, not promised:** an individual has one lived stream (a second is refused); a dream carries no store
  hash; only LIVED x SELF rows are what happened to it (`lived_rows`); `verify()` recomputes every file hash,
  manifest hash and the lived chain. `seeds_used()` makes held-out worlds auditable.

Seeds: lived from 75101; practice worker k of run r from 80,000,000 + r·1,000,000 + k·100,000; 75001–75040 are CB2
engineering evaluation; 90001–90200 stay untouched until a preregistration names them.

## 3. The predictive core

DreamerV3's RSSM (Hafner et al. 2023) from the MIT-licensed PyTorch port `NM512/dreamerv3-torch`, vendored
unchanged in `continuous_brain_cb2/third_party/dreamerv3_torch` (commit 6ef8646). It sees the frame and its body
sense, and predicts:
- the frame;
- its body;
- the reward;
- whether its life goes on;
- 12 risk labels: died, ran out of water, ran out of food, or lost 3+ health, each within 25, 100 and 300 steps
  inside the same life.

Sizes for the scaling curve: XS 6.6M, S 18.4M (DreamerV3's default), M 40M, L 85M weights. On this 2-CPU container
an S update (16 x 64 steps) takes tens of seconds, so training runs on a GPU.

## 4. The first learned skill

`SkillPolicy` acts from the core's state by imitating the routine, in two arms:
- **P-A:** core + body sense + its memory organ's readout;
- **P-B:** core + body sense only.

The gap between them measures how much it still leans on the hand-built map. The core is frozen while the policy
learns, so the probes of the core measure what the world model learned on its own.

`CoreSkillHost` binds the `skills` slot. Its weights, its recurrent state and its lineage are store channels. It
steps the core deterministically and acts greedily, so a life replays exactly from a saved state (tests).

## 5. Consolidation

`install(src, dst, weights, lineage)` carries the individual into a newly built runtime with the learned skill:
- every channel outside `skills` is kept exactly: memory, commitments, self-model, diary, speech, the clock and
  the life number;
- replacing the routine library is recorded as a *transplant*; later rounds are *installs*.

The event records:
- the state hash before and after;
- the weights' SHA-256;
- the archive shards and their manifest hashes;
- the training code and configs.

It is appended to a hash-chained `consolidations.jsonl` beside the individual. A rollback restores the snapshot
the event names.

## 6. Running it

```bash
python scripts/explore/cb2_collect.py wake --lives 20          # the individual lives; every tick archived
python scripts/explore/cb2_collect.py practice --workers 8 --steps 1000000 --run 1
python scripts/explore/cb2_sleep.py core --size S --updates 30000 --out runs/explore/cb2/sleep/r1/core
python scripts/explore/cb2_sleep.py policy --core runs/explore/cb2/sleep/r1/core/core.pt --arm P-A --out runs/explore/cb2/sleep/r1/policy-P-A
python scripts/explore/cb2_sleep.py probe --core runs/explore/cb2/sleep/r1/core/core.pt
python scripts/explore/cb2_evaluate.py install --individuals <candidate copy> --core ... --policy ...
python scripts/explore/cb2_evaluate.py evaluate --individuals <candidate copy> --label r1-PA
```

To watch the individual live its own next lives in the browser:

```bash
python scripts/explore/w1_viewer.py --individual cb2-ind-0001
```

Every tick it lives there is archived to its LIVED stream, exactly as `wake` does, and each life saves it to its
home. Ctrl+C mid-life seals the shard and saves it with its world, and the next start continues that same life
(`tests/test_cb2_viewer.py`: a life lived in the viewer, stopped and continued, is identical to one lived
headless). Erase, bookmark/back and "new individual" are refused for it: its life is its chronology.

`scripts/explore/cb2_s1.sh r1` runs all of it on a GPU machine. It installs into a *candidate copy* of the
individual and reads the gates. The canonical install is a separate, deliberate step.

Each evaluation life is archived as PRACTICE x SELF with the routine's COUNTERFACTUAL action at every step. That is
the data for the next imitation round (DAgger).

## 7. Gates (non-regression, not claims)

| gate | threshold |
|---|---|
| record/no-record twins | identical store hashes (`tests/test_cb2_archive.py`) |
| practice throughput | efficiency ≥ 0.7 × N workers (measured 1.09 at 2 workers here) |
| step = scan | exact (`tests/test_cb2_core.py`) |
| dreams never lived | the writer refuses; `lived_rows` reads LIVED x SELF only (tests) |
| consolidation | install, rollback and reinstall with matching hashes (`tests/test_cb2_consolidation.py`) |
| imitation | held-out agreement ≥ 90% on lived states (P-A) |
| closed loop | score and survival each ≥ 0.8 × the routine's, on 75001–75040 |
| probes | the core beats an untrained core and the frame encoder at recovering the 62 features and the remembered tiles out of view |

## 8. Status

- **Built and tested:** archive, wake and practice collection, core, policies, probes, skill host, consolidation,
  evaluation.
- **Collected here:** 20 lived lives (13,978 transitions) and practice run 0: two sealed shards, 131,072 transitions. The
  container was reclaimed while idle and the workers stopped partway through their second shards. Those unsealed
  shards were lost, and nothing else was: crash safety worked as designed.
- **On David's machine (RX 6600, 12 threads):** living, practice and evaluation run there. ROCm does not support the RX 6600
  under WSL, so training runs on a rented GPU.
- **Provenance note:** the lived lives and practice run 0 were collected with the CB2 code before it was first
  committed (3838fef), so their manifests name the parent commit fac701e. The recording code that produced them
  is what 3838fef contains; a later change (the shadow routine) applies only when the skill is not the routine.
- **Waiting for a GPU:** training at real scale.
