# CB2-H1 Phase 0: reading of the report (h1-phase0-report.json)

- **Run.** It ran on David's WSL at code a8507f3, on 120 fresh development worlds (77001–77120). It took about 30 min with
  6 workers. No errors, and branch fidelity held on every world (the replay and the control branch reproduced pass 1
  action for action).
- **Cross-machine check.** The first 7 worlds were also run in the container. Every pass-1 life and 41 of 42 branch
  outcomes are identical. One branch (77003, flee16: 172 vs 641 steps) differs. Within one machine everything is
  deterministic, but tiny CPU float differences can flip a branch's future. All reported numbers come from one machine
  (WSL).

## The preregistered screen passes on the letter of the rule, but the headroom evidence is weak

| Screen | Result | Threshold | Status |
|---|---|---|---|
| (a) Opportunity | 50 of 120 deaths are encounter-associated: 41.7% [32.5, 50.0] | ≥ 30% | clears |
| (b) Headroom | relative reduction 100% (4/109 → 0/109); paired lower95 of the death reduction +0.009 | ≥ 25% | clears |

Why (b) is weak:

1. **The sampled onsets are rarely lethal.**
   - Only **4 of 109** first hands-attributable onsets ended in an encounter-associated death under S2b's own hands.
   - The 100% reduction rests on 4 events.
   - Sampled onsets come early in life (median step 53, against a median life of 407).
2. **No fixed script reduces encounter damage.**
   - The paired damage differences, script vs control, are:
     - flee4: −0.07 [−0.46, 0.30]
     - flee8: +0.24
     - flee16: +0.05
     - water: +0.19
     - fight: +0.22 [+0.03, +0.43]
     - hold: +1.36 [+0.87, +1.84]
   - None of the fixed scripts' survival changes is distinguishable from 0. The best is water at +39 [−30, +110].
3. **The hindsight "best-of" survival gain (+278 [220, 335]) is mostly chaos, not escape.**
   - The 7 branch futures from one state differ by SD ≈ 214 steps, so the maximum of 7 is far above the control by
     selection alone.
   - The control itself is "best" in 17/109 cases.
   - Without a placebo calibration (perturbations that are not escapes), this number cannot be credited to escaping.

## Where the damage deaths begin (the qualification David raised)

Of the 50 damage deaths, the last encounter was:

| Last encounter | Count |
|---|---|
| hands-attributable | 13 (26%) |
| outside in the dark | 25 (14 of them also at health < 6) |
| health < 6 in daylight | 6 |
| a need ≤ 2 | 6 |

About three-quarters of the damage deaths begin weak or exposed. That points first at upstream decisions (being
outside at night, entering threats already weak), not at the hands.

## Definition fit

- Missed threats: 29.9 steps per life of 215.5 true-threat steps (14%) were not perceived.
- Hits outside any encounter: 0.63 per life.
- Timeouts: 8.6% of encounters.
- Escape (cleared) rate: 77.4%.
- Combat damage: 2.02 per encounter, 12.3 per life.
