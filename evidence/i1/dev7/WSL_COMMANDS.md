# I1 on David's WSL (Ubuntu-24.04): engineering check, then the preflight

There are two stages. **Stage A** is an ordinary engineering run on 8 individuals from `cloud-dev-claude-eng`
(906880–906887). It shows that I1 is valid on your machine before anything single-use is spent.
**Stage B** is the preflight. It registers and spends `i1-v1-preflight-eng-b` (907100–907163); that cannot
be undone, and the family can never be used again. Run Stage B only if every stop condition below is
passed.

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
git log --oneline -1
git merge-base --is-ancestor 3dda2fe HEAD && echo BASE_OK
cp ~/projects/continuous_brain_v0.38.1.dev6_composition_route/Continuous_brain/registry/seed_ranges.local.jsonl registry/
export PYTHONPATH=$PWD
python -m pytest -q -p no:cacheprovider tests/
sha256sum ~/projects/continuous_brain_v0.12.2/state/child_v113_dml_smoke/infant_training.pt
```

**Stop conditions so far:**

- **Zip hash:** it must equal the hash Claude gave you in chat.
- **`git log`:** it shows the head commit Claude gave you in chat, and the next line prints `BASE_OK` (the dev7 commits sit on top of `3dda2fe`).
- **Tests:** they must report **225 passed**.
- **Checkpoint:** the hash must be `086932388a9a410af46fea8e87cc63dd4b60161e03e247bf81fc88d28cfd577f`.

If any of these differ, stop and send Claude the terminal output.

## Stage A — engineering check (about 1–2 minutes)

```bash
python scripts/dev/i1_dev.py --legacy-root ~/projects/continuous_brain_v0.12.2 --historical-core-archive /mnt/c/Users/davei/Downloads/continuous_brain_v0.12.2_unified_workspace.zip --start 906880 --n 8 --label wsl_i1_dev8
python scripts/dev/i1_tamper_check.py runs/dev/wsl_i1_dev8 runs/dev/wsl_i1_dev8/tamper_check.json
python scripts/i1_run.py --print-design-digest
```

**Stop conditions for Stage A:**

- **The dev run** prints one line per arm (12 lines, each `"errors": 0`), then a `"validity"` block with 14
  entries, all `true`. The block ends with `"proposed_verdict_if_this_were_a_scientific_family": "PASS"` (or
  `"FAIL"`, which is still valid engineering data). Stop if:
  - it crashes before printing that block;
  - any validity entry is `false`;
  - it says `BLOCKED`.
- **The tamper check** prints one `BASE` line, then 21 `ok` lines, then `ALL DETECTED`.
- **The design digest** is `1ca74d5b44d84cc1b00be3619956881f944a387b8551938f2913cc0d7b3f37a4`. If it isn't,
  the preregistration file differs from the one reviewed.

If anything differs, don't run Stage B. Send the results with the tar command at the end.

## Stage B — preflight (spends 907100–907163; about 5–8 minutes)

```bash
python scripts/i1_preflight.py --legacy-root ~/projects/continuous_brain_v0.12.2 --historical-core-archive /mnt/c/Users/davei/Downloads/continuous_brain_v0.12.2_unified_workspace.zip --label wsl_i1_preflight
sha256sum ~/projects/continuous_brain_v0.12.2/state/child_v113_dml_smoke/infant_training.pt
```

The preflight prints one line per arm, then the 14 validity entries and the SHA-256 of the sealed report.
Nothing else is shown. The primary and secondary values are written to `runs/i1/wsl_i1_preflight/sealed/`, and
the arms' `lives.jsonl` files and plans also contain them. None of these files are encrypted, so please
don't open them, and don't send them. The tar command below sends only the preflight's value-free files.

## Send the results

```bash
tar czf /mnt/c/Users/davei/cb_exchange/wsl_i1_dev7_results.tgz --mtime='2026-01-01 00:00:00' --exclude='historical_core' --exclude='persist' runs/dev/wsl_i1_dev8 runs/dev/seed_usage.jsonl registry/seed_ranges.local.jsonl runs/i1/wsl_i1_preflight/i1_validity.json runs/i1/wsl_i1_preflight/manifest.json runs/i1/wsl_i1_preflight/seed_opened.json runs/i1/wsl_i1_preflight/safety.json runs/i1/wsl_i1_preflight/*/header.json runs/i1/wsl_i1_preflight/*.stderr.txt
```

The Stage A directory goes in whole, including its `lives.jsonl`; those are engineering values. From the
preflight, only the files named above go in. Its per-arm stdout files are left out, and every file time is set
to one fixed date, because each arm's run time grows slightly with the number of individuals that reach their
second goal. That is a weak hint at C2, but it isn't sent. Please don't read the timings in your terminal
either; Claude needs only the validity lines. If you stopped after Stage A, run the same command: tar
reports the missing preflight files and exits with code 2, but it still writes the archive.

## What to expect

- **Manifest:** it will differ from the cloud's, because the CPU, the torch build and the machine are part
  of it. The manifest your preflight writes is the one the scientific run must reproduce on this same
  machine.
- **Run time:** Stage A takes about 1–2 minutes and Stage B about 5–8, depending on the CPU.

## After the preflight (only after the review, and only if you decide to approve)

Approval is David's decision. Claude and ChatGPT review first. When you decide to approve, run this one
command on the same machine, in the same directory:

```bash
python scripts/i1_approve.py --preflight-dir runs/i1/wsl_i1_preflight --approver David --i-approve I1-INTEGRATED-INDIVIDUAL-v1
```

It refuses unless:

- the preflight is a valid, spent preflight of `i1-v1-preflight-eng-b`;
- this machine reproduces the preflight's manifest exactly.

It then writes the approval into both registry files at once:

- **`registry/i1_integration_v1.json`:** status, approved_by, approved_utc, preflight_manifest_sha256 and
  approved_design_sha256;
- **`registry/preregistrations.json`:** the I1 entry is set to APPROVED.

It runs nothing and touches no seed family. The test suite stays green after approval (225 pass in every lifecycle
state), so you can re-run it as a check at any point. The scientific run is then:

```bash
python scripts/i1_run.py --legacy-root ~/projects/continuous_brain_v0.12.2 --historical-core-archive /mnt/c/Users/davei/Downloads/continuous_brain_v0.12.2_unified_workspace.zip --confirm-open I1-INTEGRATED-INDIVIDUAL-v1
```

That spends `i1-v1-dev` (328000–328127) before launch and ends PASS, FAIL or BLOCKED. Before approval it
refuses, because the status is PROPOSED.
