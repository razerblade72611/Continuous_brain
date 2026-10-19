# Frozen Writer / New Reader Representation-Reuse Probe v1

Purpose: test whether the trained 64-D memory is reusable by a reader that did
not exist during writer training.

The original writer and recall architecture are frozen.  New probes receive
only the 64-D memory vector at the delayed decision.  They cannot access hidden,
context, image, workspace, original recall output, or motor logits.

This is a TRAIN-only representation assay, not writer generalization:
the writer itself has previously seen the TRAIN family.

Tests:
1. condition-level 5-fold CV with LEFT/RIGHT pair members kept together;
2. leave-one-corridor-length-level-out;
3. leave-one-cue-step-level-out;
4. shuffled-label negative control.

Readers:
- fresh linear 64->2 head;
- fresh MLP 64->64->2 head.

A strong result would show high held-reader-out accuracy, especially for the
linear probe and across factor-held-out reader tests, while shuffled-label
control stays near chance.

DEV and TEST remain sealed.
