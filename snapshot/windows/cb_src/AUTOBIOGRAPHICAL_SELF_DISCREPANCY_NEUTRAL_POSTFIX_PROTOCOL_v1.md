# v0.32.1a — Autobiographical Neutral Evaluator Postfix

This package is an infrastructure repair only.

The v0.32.1 training run completed and saved:

`state/autobiographical_self_discrepancy_neutral_v1/neutral_autobiographical_self_model.pt`

before the evaluator crashed.

The crash was caused by an implementation mistake:

`zero_gradient_sanity()` was decorated with `@torch.no_grad()` and then called
`backward()`.

Therefore the scientific DEV gate was never completed.

v0.32.1a:
- performs NO training;
- loads the exact already-trained v0.32.1 model;
- rebuilds the same TRAIN and reused DEV subjects deterministically;
- reruns the normal metrics;
- reruns donor/erasure/discrepancy controls;
- reruns reload checks;
- reruns the gradient sanity check with autograd enabled;
- hashes the saved v0.32.1 model and all protected artifacts before/after;
- leaves TEST subjects 6000..6255 sealed.

The v0.32.0 historical DEV FAIL remains unchanged.

If the postfixed evaluator passes, the successful status belongs to the
separately recorded v0.32.1 structural-neutrality repair.

If it fails a scientific gate, that is a real v0.32.1 DEV repair failure and
TEST remains untouched.
