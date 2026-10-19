# Continuous Brain (private)

David's project: a persistent, capable individual with the functions of a human brain, striving for sentience.
The shared goal and the agent rules are in Linear: "Continuous Brain — Shared Goal & Agent Onboarding (DRAFT)".

Pushed 2026-10-19 from David's PC (WSL `~/projects` and `C:\Users\davei\ContinuousBrain`).

## Branches (full git history)

| branch | what it is |
| -- | -- |
| `explore/l1b-frozen-53363a4` | **newest code**: the frozen L1b revision that ran on RunPod Oct 12–13 (53363a4). Includes 4c7eec9 and e4a82d3. From Sol's review clone `~/cb_data/sol-l1b-review-20261011` |
| `explore/cb2-l1fix` | L1b bench controller line (f1369cd, Oct 11) |
| `explore/cb2-latest` | latest committed CB2 work (a8507f3, Oct 10: H1 Phase 0 census, DAV-97). Detached HEAD of `~/projects/continuous_brain_explore` |
| `explore/cb2-r3` | CB2 r3 line, ending at the canonical install of r3h-B (5668688, Oct 9) |
| `explore/w1-world-and-organs` | W1 world, vision and language organs (Oct 4–7) |
| `history/e0-snn` | Sept E0 Genesis SNN code (882e584), previously the only content on GitHub |

## `main` = file snapshot

- `snapshot/projects/` — every folder under WSL `~/projects` (InfantBrain versions v0.1–v0.12.2, author backups, v0.38.x, cb_src, continuous_brain_explore working tree including uncommitted files, ai2thor-humanoid source).
- `snapshot/windows/` — `C:\Users\davei\ContinuousBrain` (cb_exchange agent channel incl. L1b reviews, mirrors, cb_github, cb_explore_src).

Included: source and text files under 2 MB (.py .md .json .jsonl .yaml .toml .sh .ps1 .ipynb .csv and similar).

Left out on purpose:
- model weights and checkpoints (.pt .pth .ckpt .safetensors .npz), saved `checkpoints/` and `lives/` world states, archives (.tgz .zip), files over 2 MB;
- `.venv`, `__pycache__`, `node_modules`, `.git` internals;
- `llama.cpp` (third-party, available upstream);
- anything named `.env*`, `*.pem`, `id_*`. A key scan found no credentials;
- whatever each project's own `.gitignore` excludes (about 140 MB: `state/`, `runs/`, `registry/`, `build/`, caches, egg-info). Those stay on David's PC.

`LEFT_OUT_CHECKPOINTS.sha256` lists every left-out checkpoint and large file with its SHA-256, so the copies on David's PC can be matched later.
