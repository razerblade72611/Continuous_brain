# RunPod use

v0.38.0.dev0 does not automatically allocate cloud resources.

Recommended workflow:

1. test locally;
2. package the exact commit/config;
3. choose a pod only after the experiment has a purpose and stop rule;
4. record GPU model, wall time, peak memory and cost estimate in the run manifest;
5. prefer replication/falsification over model size.

Use `scripts/prepare_runpod.sh` to print a setup recipe. It performs no network or billing action.
