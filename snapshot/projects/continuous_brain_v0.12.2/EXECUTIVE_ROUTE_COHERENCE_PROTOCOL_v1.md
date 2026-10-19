# Executive Route-Coherence Diagnostic v1

This diagnostic follows the formal v0.13.2 fresh-DEV admission FAIL.

It does not alter the original thresholds and does not reclassify that FAIL.

Observed before this diagnostic:

- fresh DEV pair-full:
  - invert_lr = 0.79
  - vertical = 0.89
  - vertical_reverse = 0.91
- executive OFF = 0 for all active rules;
- executive memory ZERO = 0 for all active rules;
- executive-only donor swap:
  - invert_lr = 1.00
  - vertical = 0.00
  - vertical_reverse ≈ 0.00.

Hypothesis:

The executive-only donor intervention may create an internally inconsistent
current-step cue state. The admitted direct recall route D still receives the
recipient's original memory while the executive E receives donor memory.

At fixed recipient hidden state, independently control:

- H = historical current-memory input
- D = admitted direct memory→workspace recall
- E = novel executive memory input

Primary interpretation:

- If DE_DONOR or HDE_DONOR restores donor consistency for vertical rules,
  the failed donor test mainly exposed route incoherence.
- If coherent donor swaps also fail, vertical rule execution depends on a
  deeper recipient-state interaction and should not be described as a portable
  current-memory substitution.

No optimizer steps.
Fresh TEST remains sealed.
