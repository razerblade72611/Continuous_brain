# Continuous Brain v0.9.2 — Embodied Evidence Hotfix

v0.9.2 preserves the v0.9 child lineage and fixes two label-leakage issues before world-model/motor training:

- separates parent motor target (`action`) from the action actually executed (`executed_action`)
- conditions the world model on the executed action that caused the recorded transition
- stores and uses a true `previous_action` token instead of feeding the current motor target back into the nucleus
- trains every world-model ensemble member, making ensemble disagreement meaningful
- gives the temporal cortex a grounded single-transition prediction objective
- adds held-out world-model metrics: no-change baseline, wrong-action control, reward/success, uncertainty
- adds `merge-nurseries` so static parent lessons and fresh lived transitions can be replayed together
- corrects nursery backup filename: v0.9 nurseries use `lessons.jsonl`

No v0.9.1 child weights need to be discarded. Resume the same checkpoint after installing v0.9.2.
