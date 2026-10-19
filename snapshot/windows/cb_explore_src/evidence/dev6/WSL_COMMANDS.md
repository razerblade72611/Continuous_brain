# Running the dev6 checks on David's WSL (Ubuntu-24.04)

These checks are ENGINEERING_ONLY. They use only the registered engineering family `cloud-dev-claude-eng` (906000–906999). They read the legacy tree and never write to it; every run checks the tree metadata before and after, and hashes the protected checkpoint. Nothing here registers or opens a G1 family.

Run everything in the terminal where `(cb_v0381_g0)` is active. Windows (PowerShell): `wsl -d Ubuntu-24.04`. Then in WSL:

```bash
cd ~/projects
mkdir -p continuous_brain_v0.38.1.dev6_composition_route
cd continuous_brain_v0.38.1.dev6_composition_route
sha256sum /mnt/c/Users/davei/cb_exchange/Continuous_brain_v0.38.1.dev6_composition_route.zip
unzip -q /mnt/c/Users/davei/cb_exchange/Continuous_brain_v0.38.1.dev6_composition_route.zip
cd Continuous_brain
git log --oneline -3
cp ~/projects/continuous_brain_v0.38.1.dev5_g1_prereg_v2/Continuous_brain/registry/seed_ranges.local.jsonl registry/
export PYTHONPATH=$PWD
python -m pytest -q -p no:cacheprovider tests/
sha256sum ~/projects/continuous_brain_v0.12.2/state/child_v113_dml_smoke/infant_training.pt
python scripts/dev/e1_null_composition.py --legacy-root ~/projects/continuous_brain_v0.12.2 --historical-core-archive /mnt/c/Users/davei/Downloads/continuous_brain_v0.12.2_unified_workspace.zip --n 8 --tag wsl_e1n8
python scripts/dev/compare_cross_env.py runs/dev/wsl_e1n8_e1_report.json
sha256sum ~/projects/continuous_brain_v0.12.2/state/child_v113_dml_smoke/infant_training.pt
tar czf /mnt/c/Users/davei/cb_exchange/wsl_dev6_results.tgz runs/dev/wsl_e1n8_e1_report.json runs/dev/wsl_e1n8_full runs/dev/wsl_e1n8_used runs/dev/seed_usage.jsonl
```

What to expect:

- **Zip hash:** the first `sha256sum` matches the hash Claude gave you in chat.
- **Commit:** `git log` shows the dev6 commit on top of `3e8459a`.
- **Tests:** 205 pass.
- **Protected checkpoint:** both checkpoint hashes are `086932388a9a410af46fea8e87cc63dd4b60161e03e247bf81fc88d28cfd577f`.
- **E1:** prints `E1_all PASS` (about 2–4 minutes).
- **Cross-environment comparison:** `compare_cross_env.py` prints `CROSS_ENV_ALL PASS` if your machine reproduces the cloud result files byte for byte. A FAIL there is information, not damage. It would mean cloud development results do not transfer bit-exactly to your machine, and engineering gates must then be re-run on WSL.

`PYTHONPATH=$PWD` makes this directory's code take precedence for this terminal only. Your dev5 checkout and its editable install stay untouched.
