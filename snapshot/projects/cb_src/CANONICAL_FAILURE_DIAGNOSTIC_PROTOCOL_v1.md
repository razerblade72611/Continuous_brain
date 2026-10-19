# Canonical v0.14.0 Failure Diagnostic v1

No model training. No TEST access.

The fresh v3 DEV admission failed, but the training trace remained visibly
unstable through the end of the one-epoch run. Therefore we must not attribute
the failure solely to representation generalization without measuring the
fixed-final TRAIN policy.

This diagnostic answers:

1. Does the update-1600 canonical executive actually solve all 1600 TRAIN
   structural conditions at fixed final weights?

2. Is cue identity still linearly portable from v3 TRAIN to v3 DEV in:
   - raw working memory64;
   - canonical recall residual128;
   - recurrent hidden state?

Interpretation:

- weak TRAIN:
  optimization/underlearning is primary; architecture conclusions are premature.

- strong TRAIN + weak DEV:
  genuine held-out generalization gap.

- raw memory64 strong but canonical recall128 weak:
  canonical recall transform loses or entangles useful portable content.

- both raw memory64 and canonical recall128 strong but executive DEV weak:
  content survives; rule-conditioned executive transformation/generalization is
  the bottleneck.

The already-opened v3 DEV representations may be evaluated, but no reader is
fit on DEV. Probe readers are fit only on v3 TRAIN.

Fresh TEST remains sealed.
