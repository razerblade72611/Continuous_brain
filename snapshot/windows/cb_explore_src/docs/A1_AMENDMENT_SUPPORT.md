# A1 Comparator / Parity Amendment Support — PROPOSED

This code supports a proposed refinement. It does **not** silently amend the historical A1 draft and it does not create a scientific result.

## Why the literal total-parameter match is insufficient by itself

For the canonical E0 seed-0 network used during this implementation review:

- all active SNN synapses: **375,231**
- active cortex→cortex SNN synapses: **343,528**

With the shared 2,182-channel physical encoder, the literal A1 total-learned-scalar match gives:

- `DENSE-TOTAL`: GRU hidden size **42**
- total learned parameters: **374,302**
- recurrent GRU weights (`weight_hh`): **5,292**

This satisfies the literal total-budget rule but spends almost all of the dense budget on input/output mappings, not recurrent state transition capacity.

## A1.1b proposal: report both comparator interpretations

Keep `DENSE-TOTAL` as a report-only implementation of the original draft, and add:

- `DENSE-CORE`: match recurrent GRU `weight_hh` scalars (`3 H^2`; recurrent bias excluded from the core count) to active SNN cortex→cortex synapses within ±10%.

At the same canonical E0 numbers:

- `DENSE-CORE` hidden size: **338**
- recurrent weights: **342,732**
- total learned parameters: **3,297,006**

Input projection, prediction head, total parameters, wall time, accelerator time and peak memory remain separately reported. The auxiliary prediction head is not an E2 probe feature.

## A1.2b proposal: familywise parity cutoff

If both firing-rate and CV checks are treated as familywise claims for each of 140 populations, the family has 280 hypotheses.

The two-sided Bonferroni normal cutoff is:

`z = 3.7475483420041256`

The implementation exposes this as the proposed 280-test rule. The original 140-test `z = 3.57` rule remains available and distinguishable in configuration/history.

No cutoff is to be changed after parity data are inspected.
