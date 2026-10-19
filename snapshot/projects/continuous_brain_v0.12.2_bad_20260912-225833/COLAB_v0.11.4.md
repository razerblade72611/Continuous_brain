# Colab v0.11.4

Open `Continuous_Brain_v0.11.4_PROTECTED_GROWTH_COLAB.ipynb`, select a GPU runtime,
and choose **Runtime → Run all**. See `START_HERE_v0.11.4.md` for inputs and outputs.

The notebook uploads/verifies the source ZIP and checksum, reuses the exact
v0.11.0 bundle and sealed world assets, stages active work on local SSD, warms
training data into host page cache, runs CUDA with FP32 and batch 8, and verifies
periodic Drive checkpoint copies every 250 updates. It never fills VRAM merely
to increase the utilization display.

A clean protected branch is created under
`MyDrive/continuous_brain_v114/state/child_v114_protected_r2/`.
Outputs are under
`MyDrive/continuous_brain_v114/results/protected_integration_r2/`.

The full protocol is baseline P4b; 2,000 shadow, 1,000 micro, 1,000 low, and 1,000
moderate updates with P4b after each; bootstrap/world retention; randomized
DEVELOPMENT; measured readiness. Frozen stage endpoints and runtime-bound caches
support repeat execution. A failed stage stops advancement.

No final generalized blind evaluation is included. The notebook has been syntax
checked and its orchestration logic tested locally; no user's Drive session or
A100 qualification was executed during release construction.
