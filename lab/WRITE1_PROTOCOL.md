# WRITE-1 protocol (frozen before data, 2026-10-27)

Designed by Claude and Sol after DECAY-1. The frozen code is authoritative:
- `scripts/write1.py` (runner) and `scripts/write1_analyze.py` (rules);
- tests: `tests/test_write1.py` (unit) and `tests/e2e_tiny_write1.py` (the real runners on a tiny CPU backbone; all
  reproductions and integrity checks exact).

## Question
Can a new life be written without changing the parameters that hold earlier memories, and does the model then still
answer the old questions when both are active, with no routing and no life label?

## Design
- **Seeds and streams:** seeds 2–5, BIND-I1 streams.
- **Life 1:** re-run exactly; it must reproduce BIND-I1.
- **Continuation:** one life on the U-type new facts, branched from the full checkpoint-1 state. The KL-to-frozen
  objective, schedule, facts and probes are identical across arms.

| Arm | Life-2 write |
|---|---|
| A | continue the life-1 adapter, inherited AdamW (must reproduce BIND-I1 U life 2 and DECAY-1 arm A exactly) |
| R | continue the life-1 adapter, fresh AdamW (optimizer-reset control) |
| P | life-1 adapter frozen and active; a fresh rank-16 adapter (zero initial output, seeded) is the only trainable memory, with fresh AdamW |

- **P readouts:**
  - combined (primary);
  - old-only (integrity gate: equals checkpoint 1 by construction);
  - new-only (descriptive).
- **P integrity:**
  - the combined readout before life-2 training must equal checkpoint 1;
  - the frozen-KL teacher disables both adapters;
  - trainable-parameter counts and steps must match across arms.

## Rules
- **Gates:** V1, V2, V3, R (three exact reproductions) and I (P integrity). Any failure makes the run INVALID.
  V3b and V5 are qualifiers.
- **Qualifying seed:** at life 2, old-target forced choice ≥ 3/4 and binding ≥ 3/4 in the same seed.
- **Preserved:** ≥ 3 of 4 seeds qualify.
- **Learned:** U-fact uptake ≥ 3/4 in ≥ 3 of 4 seeds.
- **P's combined readout** is classified as:

  | Old memories | New facts | Interpretation |
  |---|---|---|
  | Preserved | Learned | Isolated incremental memory works for one life |
  | Not preserved | Learned | Readout interference demonstrated |
  | Preserved | Not learned | Stability bought with plasticity |
  | Not preserved | Not learned | Fails both; cause unresolved |

- **Contrasts:** P−A, R−A and P−R on dV and dB, with the 2-nat / 3-of-4 rule.
  - R−A robust_pos with P−R not robust_pos → "optimizer reset deserves investigation".
- **Secondary:** counter-entity forced choice, value intrusion d l(U|A), the new-only readout, and A/R classified the
  same way.
- **Pre-registered predictions:**
  - Claude: P combined beats A modestly but fails preservation.
  - Sol: the old adapter is protected internally but the combined output stays vulnerable.
