# Working Memory Recall — Final TRAIN Causal Gate v1

DEV and TEST remain sealed.

The 1600-update Phase-1 audit contains online measurements collected while each model was still learning. It is valid evidence about learning dynamics, but it is not a fixed final-policy comparison.

This gate evaluates the final update-1600 M1.5 and M2 checkpoints on all 1600 TRAIN conditions at fixed weights and performs matched delayed-decision interventions.

Primary interventions:
- normal final policy;
- direct recall disabled while historical memory remains intact;
- direct-recall memory swapped across LEFT/RIGHT while historical memory stays recipient-owned;
- total memory swapped;
- hidden reset + historical-memory-only;
- hidden reset + direct-recall-only;
- hidden reset + direct-recall swap;
- hidden reset + both memory paths;
- hidden reset + all memory zero (LEFT/RIGHT logits must be identical).

Primary causal questions:
1. Does direct workspace recall add full-action and side-sign value beyond the inherited memory route?
2. Does swapping only direct-recall content push logits toward the donor cue?
3. After recurrent hidden is erased and the inherited current-memory input is zeroed, can direct recall alone decode own cue memory, and does swapped memory reverse the donor direction?
4. Does this causal behavior survive long realized hidden delays?

Only TRAIN is used. A successful result may justify a parameter-matched ungated capacity control or freezing a candidate architecture for DEV, but it is not DEV evidence itself.
