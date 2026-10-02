# I1 on David's WSL (Ubuntu-24.04): engineering check, then the preflight

There are two stages. **Stage A** is an ordinary engineering run on 8 individuals from `cloud-dev-claude-eng`.
It shows that I1 is valid on your machine before anything single-use is spent. **Stage B** is the preflight.
It registers and spends `i1-v1-preflight-eng-b` (907100–907163); that cannot be undone, and the family can
never be used again. Run Stage B only if Stage A ends with every validity check `true` and `ALL DETECTED`.

Nothing here opens the scientific family `i1-v1-dev`, and nothing writes to the legacy tree. Every run
compares the tree's metadata before and after, and you hash the protected checkpoint before and after.

**Rule: don't edit, copy into or pull into the `Continuous_brain` directory while a run is going.** Every
arm records a hash of all the code, and a change in the middle of a run blocks it. (This happened once in
the cloud; see `docs/I1_INTEGRATED_INDIVIDUAL_DEV7.md` §5.)

Open WSL from PowerShell with `wsl -d Ubuntu-24.04`, activate `(cb_v0381_g0)`, then:

```bash
cd ~/projects
mkdir -p continuous_brain_v0.38.1.dev7_i1
cd continuous_brain_v0.38.1.dev7_i1
sha256sum /mnt/c/Users/davei/cb_exchange/Continuous_brain_v0.38.1.dev7_i1.zip
unzip -q /mnt/c/Users/davei/cb_exchange/Continuous_brain_v0.38.1.dev7_i1.zip
cd Continuous_brain
git log --oneline -3
cp ~/projects/continuous_brain_v0.38.1.dev6_composition_route/Continuous_brain/registry/seed_ranges.local.jsonl registry/
export PYTHONPATH=$PWD
python -m pytest -q -p no:cacheprovider tests/
sha256sum ~/projects/continuous_brain_v0.12.2/state/child_v113_dml_smoke/infant_training.pt
```

## Stage A — engineering check (about 1–2 minutes)

```bash
python scripts/dev/i1_dev.py --legacy-root ~/projects/continuous_brain_v0.12.2 --historical-core-archive /mnt/c/Users/davei/Downloads/continuous_brain_v0.12.2_unified_workspace.zip --start 906820 --n 8 --label wsl_i1_dev8
python scripts/dev/i1_tamper_check.py runs/dev/wsl_i1_dev8 runs/dev/wsl_i1_dev8/tamper_check.json
```

**Before going on, check the output of both commands:**

- **The dev run prints a `"validity"` block.** All 12 entries must be `true`. The block ends with
  `"proposed_verdict_if_this_were_a_scientific_family": "PASS"` (or `"FAIL"`, which is still valid
  engineering data). If any validity entry is `false`, or it says `BLOCKED`, **stop**: don't run Stage B.
  Send the Stage A results with the tar command at the end and I'll diagnose.
- **The tamper check** prints 16 `ok` lines and ends with `ALL DETECTED`.

## Stage B — preflight (spends 907100–907163; about 5–8 minutes)

```bash
python scripts/i1_preflight.py --legacy-root ~/projects/continuous_brain_v0.12.2 --historical-core-archive /mnt/c/Users/davei/Downloads/continuous_brain_v0.12.2_unified_workspace.zip --label wsl_i1_preflight
python scripts/i1_run.py --print-design-digest
sha256sum ~/projects/continuous_brain_v0.12.2/state/child_v113_dml_smoke/infant_training.pt
```

The preflight shows **validity only**. The primary and secondary values are written to
`runs/i1/wsl_i1_preflight/sealed/` and committed by SHA-256. That file is not encrypted, so please don't open
it, and don't send it. The tar command below leaves it out, along with every `lives.jsonl`, since those
contain the values too.

## Send the results

```bash
tar czf /mnt/c/Users/davei/cb_exchange/wsl_i1_dev7_results.tgz --exclude='historical_core' --exclude='persist' runs/dev/wsl_i1_dev8 runs/dev/seed_usage.jsonl registry/seed_ranges.local.jsonl runs/i1/wsl_i1_preflight/i1_validity.json runs/i1/wsl_i1_preflight/manifest.json runs/i1/wsl_i1_preflight/seed_opened.json runs/i1/wsl_i1_preflight/safety.json runs/i1/wsl_i1_preflight/*/header.json runs/i1/wsl_i1_preflight/*.stdout.txt runs/i1/wsl_i1_preflight/*.stderr.txt
```

If you stopped after Stage A, run the same command; tar warns about the missing preflight files, and that
is fine.

## What to expect

- **Tests:** 218 pass.
- **Protected checkpoint:** all hashes are `086932388a9a410af46fea8e87cc63dd4b60161e03e247bf81fc88d28cfd577f`.
- **Manifest:** it will differ from the cloud's `230aa4d0…`. That is expected, because the CPU, the torch
  build and the machine are part of it. The manifest your preflight writes is the one the scientific run
  must reproduce on this same machine.
- **Design digest:** `--print-design-digest` prints `22950124bda6044079fcde32457aeb730e247cf6c14210ddaab6f3a3472dc455`.
  If it prints anything else, the preregistration file differs from the one reviewed.

## After the preflight (not yet: needs the review first)

Approval is David's decision, made after ChatGPT has reviewed the proposal. It means editing
`registry/i1_integration_v1.json` to set:

- `status` to `"APPROVED"`;
- `approved_by` to `"David"`;
- `preflight_manifest_sha256` to the value in `runs/i1/wsl_i1_preflight/manifest.json`;
- `approved_design_sha256` to the printed design digest.

Claude will prepare that edit as an exact command once the preflight result is in. The scientific run is then:

```bash
python scripts/i1_run.py --legacy-root ~/projects/continuous_brain_v0.12.2 --historical-core-archive /mnt/c/Users/davei/Downloads/continuous_brain_v0.12.2_unified_workspace.zip --confirm-open I1-INTEGRATED-INDIVIDUAL-v1
```

Don't run this before the approval. It refuses anyway: the status is PROPOSED, and no manifest or design
digest has been recorded.
