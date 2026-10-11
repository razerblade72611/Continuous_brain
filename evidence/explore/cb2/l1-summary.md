# CB2-L1 result (DAV-98) -- 2026-10-11

Run: code b6a76ee, launched 2026-10-11 03:00 UTC on 2x RTX 4090, 10 wake/sleep cycles + final probes, finished 19:25 UTC.
Results archive cb2_l1_results.tgz sha256 a056d611...209daf (downloaded, checksum matched). Report recomputed locally: identical to the pod's.

Integrity: cycle 0 (before any sleep) paired differences exactly 0 (instrument check passed). All 20 sleeps clean:
chooser value loss finite, 0 skipped updates, gradient norm < 2 (clip at 100 never engaged). No tripwire fired.
Earlier attempts (OOM hang, GPU device bug, value-head NaN, contamination by orphaned processes) produced no data used here.
Caveat: GPU nondeterminism made the learning and reference twins differ slightly after sleep 1 (design expected identical).

## Preregistered primary claims (pooled cycles 6-10, one-sided 95% bootstrap lower bound, 200 paired lives)

| comparison | fresh mean (lower95) | transfer mean (lower95) | claim met? |
|---|---|---|---|
| learning - reference ("own experience causes improvement") | +44.1 (-4.7) | +53.1 (+5.8) | no (fresh fails) |
| learning - frozen ("learning helps at all") | +52.8 (+4.3) | -9.6 (-62.1) | no (transfer fails) |
| situations, vs reference / vs frozen | -2.2 (-11.0) / -0.1 (-13.4) | | no |

Preregistered reading: **not shown**.
Change-of-direction triggers: the pattern "advantage on fresh but not transfer (vs frozen)" occurred -- the preregistered rule
reads it as memorisation, but this run does not establish memorisation as the cause (Sol's audit, correctly);
"learned-authored primitive share stays 0% -> L2 required" FIRED; "no fresh advantage vs frozen -> redesign" did NOT fire;
"chooser credited 0" did NOT fire.

## Secondary
- Gate 2 (cycle 10, learning twin chooser on - off): fresh +133 (lower95 +53), transfer +57.5 (-39.8), situations -5.6. Pooled mean > 0: chooser credited.
- Chooser-changed share of decisions: 0 at cycle 0 (no chooser yet), 2.6-6.1% after.
- Retention (situations, c10 - c5): +7.7 (lower95 -11.1 > -25): no forgetting detected.
- Gate 4 (learning twin core after cycle 10 vs frozen core): known-out-of-view R^2 0.093 vs 0.019 (frame 0.024, untrained -0.011);
  map head out of view: balanced accuracy 0.373 vs chance 0.108 (seen accuracy 0.733 vs constant 0.629).
  CORRECTION (Sol's audit): the probe's test lives were in the core's training replay pool (not held out from core training),
  so this is descriptive only; Gate 4 stays OPEN pending a genuinely held-out test. Also not compared against the reference twin.
- Hands model (primitive-action prediction) training accuracy: learning 0.95-0.99, reference ~1.00 (fits its fixed 20-life buffer).
- Learned-authored primitive share: 0% by construction.
- Growth curve: per-cycle paired differences swing by +/-100 steps with 40 worlds; no clear trend.

## Reading (beyond the preregistered verdict, labelled as interpretation)
- The reference twin (sleeping on a fixed 20-life buffer) did worse than frozen on transfer; learning - reference on transfer is largely
  the reference's degradation, not the learning twin's growth. Accumulating experience avoided that harm and gave a modest fresh gain.
- The learned chooser contributes (Gate 2) but its benefit does not carry to transfer worlds overall.
- Noise is the binding constraint: single run, 40 worlds per probe family per cycle, very wide per-cycle intervals.

## Corrections and defects found by Sol's independent audit (2026-10-11, C:/Users/davei/Documents/Codex/.../Continuous_Brain_L1_Audit.md)
The verdict (not shown; keep S2b) stands; the audit reproduced every primary number. Defects in my implementation:
1. Terminal rewards lost at life boundaries in chooser training (previous-reward input masked on a new life's first row, then
   shifted): all 200 terminal rewards (~-305 total) were dropped; the -10 death signal remained.
2. Life IDs collide across cycles (cycle and worker offsets both 1000): 38 distinct IDs for 200 worlds. Primary comparisons pair
   by world seed and are unaffected; life-ID-based holdouts are broken.
3. Gate 4 not held out from core training (above).
4. Replay excludes streams shorter than T=256 (learning: 20 worlds, 3,620 transitions) -- early deaths underrepresented.
5. "dusk_outside" checks darkness and wakefulness, not being outside shelter.
NaN mechanism reproduced locally by Sol (old code fails at update 178 on real cached batches; fixed code completes 1,500).
