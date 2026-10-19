# AUTHOR v0.8.0 — Operator README

This guide covers fresh startup, llama.cpp, v0.8.0 installation, clean/continued headless training, normal two-agent simulation, learned-state management, manipulation/texting tests, logs, tuning, troubleshooting, and Unity humanoid rebuilds.

## 1. Important paths

```text
WSL project:        /home/davei/projects/author
Python venv:        /home/davei/.venvs/author
Persistent state:   /home/davei/projects/author/state
Custom AI2-THOR:    /home/davei/projects/ai2thor-humanoid/builds/thor-humanoid-Linux64/thor-humanoid-Linux64
Dashboard:          http://localhost:8765
Windows llama.cpp:  E:\DaveInc\llama.cpp
llama.cpp port:     8080
```

## 2. Fastest normal startup from a powered-on machine

### Windows PowerShell — start llama.cpp

```powershell
cd E:\DaveInc\llama.cpp

.\llama-server.exe `
  -m "E:\DaveInc\OllamaModels\gguf_files\Qwen2-VL-2B-Instruct-Q4_K_M.gguf" `
  --mmproj "E:\DaveInc\OllamaModels\gguf_files\mmproj-Qwen2-VL-2B-Instruct-f16.gguf" `
  --alias qwen2-vl-2b `
  -ngl 999 `
  -c 16384 `
  --host 0.0.0.0 `
  --port 8080
```

Leave that PowerShell window open.

### Windows PowerShell — enter WSL

```powershell
wsl -d Ubuntu-24.04
```

### WSL — enter the project and activate Python

```bash
cd ~/projects/author
source ~/.venvs/author/bin/activate
```

### WSL — point AUTHOR at the Windows llama.cpp server

```bash
WINDOWS_HOST=$(ip route show | awk '/default/ {print $3; exit}')
export AGENT_LLM_BASE_URL="http://$WINDOWS_HOST:8080/v1"
export AGENT_LLM_MODEL="qwen2-vl-2b"

echo "$AGENT_LLM_BASE_URL"
curl -s "$AGENT_LLM_BASE_URL/models" | python -m json.tool
```

Do not launch the social simulation until the curl command returns the model list.

### WSL — launch the normal two-agent session

```bash
GALLIUM_DRIVER=d3d12 \
MESA_D3D12_DEFAULT_ADAPTER_NAME=AMD \
python -m agent_core.cli \
  --scene FloorPlan1 \
  --session \
  --agents 2 \
  --llm \
  --vision \
  --interact \
  --dashboard \
  --vision-every 1 \
  --step-delay 0 \
  --roam-decisions 500 \
  --max-steps 3000 \
  --thor-executable /home/davei/projects/ai2thor-humanoid/builds/thor-humanoid-Linux64/thor-humanoid-Linux64
```

Open in Windows:

```text
http://localhost:8765
```

## 3. Commands available inside the running social session

```text
/a <task>       Give Agent A an explicit task.
/b <task>       Give Agent B an explicit task.
/both <task>    Give both agents the same explicit task.
/a roam         Return A to autonomous roaming.
/b roam         Return B to autonomous roaming.
/both roam      Return both to autonomous roaming.
/status         Show current session status.
/quit           Shut the session down cleanly.
```

Examples:

```text
/a Pick up the apple if you can see and reach it.
/b Open the refrigerator if you decide it is useful.
/a Text B and tell B what you found near the counter.
/both roam
/status
/quit
```

Texting is an LLM capability (`TextPeer`), not a special console command. Giving A a task to text B lets A choose/use that capability.

## 4. Headless accelerated training — clean first run

Training does not require llama.cpp. v0.8.0 makes zero LLM calls in the accelerated trainer.

### WSL

```bash
cd ~/projects/author
source ~/.venvs/author/bin/activate

RESET_MEMORY=1 \
EPISODES=1000 \
WORKERS=1 \
./scripts/run_training_wsl.sh --headless
```

At startup verify:

```text
"headless": true
"llm_calls": 0
```

`RESET_MEMORY=1` deletes only the configured persistent state directory before training. Do not use it on continuation runs unless you intentionally want to erase all learned state.

## 5. Continue training without erasing learning

```bash
cd ~/projects/author
source ~/.venvs/author/bin/activate

EPISODES=5000 \
WORKERS=2 \
./scripts/run_training_wsl.sh --headless
```

`EPISODES=5000` means 5,000 total episodes. With two workers, the trainer splits those episodes between the workers and merges their experience afterward.

## 6. Training knobs

### Episode count

```bash
EPISODES=1000 WORKERS=1 ./scripts/run_training_wsl.sh --headless
EPISODES=5000 WORKERS=2 ./scripts/run_training_wsl.sh --headless
EPISODES=20000 WORKERS=2 ./scripts/run_training_wsl.sh --headless
```

More episodes = more experience and longer runtime.

### Worker count

```bash
EPISODES=5000 WORKERS=1 ./scripts/run_training_wsl.sh --headless
EPISODES=5000 WORKERS=2 ./scripts/run_training_wsl.sh --headless
EPISODES=5000 WORKERS=4 ./scripts/run_training_wsl.sh --headless
```

Each worker owns a separate Unity process. More workers are not guaranteed to be faster once CPU/GPU contention begins. Benchmark 1, then 2, before increasing further.

### Peer-encounter fraction

Default is 0.25 (25%).

```bash
EPISODES=5000 WORKERS=2 \
./scripts/run_training_wsl.sh --headless \
  --peer-encounter-rate 0.50
```

Allowed range is 0.0 through 1.0.

### Peer-encounter success radius

Default is 1.25 meters.

```bash
./scripts/run_training_wsl.sh --headless \
  --peer-encounter-distance 1.0
```

### Maximum primitive steps per episode

Default is 120.

```bash
./scripts/run_training_wsl.sh --headless \
  --max-steps-per-episode 200
```

Higher values give difficult routes more time but can lengthen bad episodes.

### Training render resolution

Default is 160x120.

```bash
./scripts/run_training_wsl.sh --headless --width 160 --height 120
```

For the current navigation trainer, low resolution reduces rendering overhead. Keep 160x120 as the tested baseline.

### Reproducible random seed

Default seed is 72611.

```bash
./scripts/run_training_wsl.sh --headless --seed 72611
```

Use the same seed when comparing configuration changes.

### Alternate persistent-state directory

```bash
STATE_DIR=state_experiment_01 \
EPISODES=1000 \
WORKERS=1 \
./scripts/run_training_wsl.sh --headless
```

This is useful for A/B experiments without destroying the main `state/` directory.

### Alternate custom AI2-THOR executable

```bash
THOR_EXE=/path/to/another/thor-player \
./scripts/run_training_wsl.sh --headless
```

## 7. Dry-run a training plan without launching Unity

```bash
cd ~/projects/author
source ~/.venvs/author/bin/activate

EPISODES=5000 \
WORKERS=2 \
./scripts/run_training_wsl.sh \
  --headless \
  --peer-encounter-rate 0.50 \
  --max-steps-per-episode 150 \
  --dry-run
```

This validates and prints the resolved plan without starting simulation.

## 8. Direct training command without the helper script

```bash
cd ~/projects/author
source ~/.venvs/author/bin/activate

GALLIUM_DRIVER=d3d12 \
MESA_D3D12_DEFAULT_ADAPTER_NAME=AMD \
python -m agent_core.training \
  --config config.yaml \
  --scene FloorPlan1 \
  --episodes 1000 \
  --workers 1 \
  --headless \
  --width 160 \
  --height 120 \
  --peer-encounter-rate 0.25 \
  --peer-encounter-distance 1.25 \
  --max-steps-per-episode 120 \
  --state-dir state \
  --thor-executable /home/davei/projects/ai2thor-humanoid/builds/thor-humanoid-Linux64/thor-humanoid-Linux64
```

Add `--reset-memory` only for a deliberately clean training lineage.

## 9. Clear memory / learned state

### Safe standalone reset

```bash
cd ~/projects/author
source ~/.venvs/author/bin/activate
python scripts/reset_training_memory.py --state-dir state --yes
```

### Clean training run

```bash
RESET_MEMORY=1 EPISODES=1000 WORKERS=1 \
./scripts/run_training_wsl.sh --headless
```

### Clear state before a normal simulation

Supported, but usually not recommended after training:

```bash
python -m agent_core.cli \
  --session \
  --agents 2 \
  --reset-memory \
  ...
```

This erases map, checkpoint, replay, persistent memory, and peer texts in the configured state directory before the run.

## 10. Back up and restore learned state

### Backup

```bash
cd ~/projects/author
cp -a state "state_backup_$(date +%Y%m%d_%H%M%S)"
```

### List backups

```bash
ls -ld state_backup_*
```

### Restore a chosen backup

```bash
cd ~/projects/author
rm -rf state
cp -a state_backup_YYYYMMDD_HHMMSS state
```

Replace the placeholder with the real backup directory name.

## 11. Inspect what training learned

### Training summary

```bash
cd ~/projects/author
python -m json.tool state/metrics/training_summary.json | less
```

### Training history

```bash
tail -n 20 state/metrics/training_history.jsonl
```

### Number of replay transitions

```bash
wc -l state/replay/navigation_replay.jsonl
```

### Persistent map

```bash
python -m json.tool state/maps/FloorPlan1.json | less
```

### Navigation checkpoint

```bash
python -m json.tool state/checkpoints/navigation_skill.json | less
```

### Agent A memory

```bash
tail -n 50 state/memory/a/memory.jsonl
```

### Agent B memory

```bash
tail -n 50 state/memory/b/memory.jsonl
```

### Peer text transcript

```bash
tail -n 50 state/messages/peer_texts.jsonl
```

Watch peer texts live:

```bash
tail -f state/messages/peer_texts.jsonl
```

## 12. Test real object manipulation

```bash
cd ~/projects/author
source ~/.venvs/author/bin/activate

GALLIUM_DRIVER=d3d12 \
MESA_D3D12_DEFAULT_ADAPTER_NAME=AMD \
python scripts/smoke_test_interactions.py \
  --scene FloorPlan1 \
  --thor-executable /home/davei/projects/ai2thor-humanoid/builds/thor-humanoid-Linux64/thor-humanoid-Linux64
```

The test attempts real grounded pickup/drop and opportunistically open/toggle actions without `forceAction=True`.

## 13. Test the two-agent custom Unity player without the LLM

```bash
cd ~/projects/author
source ~/.venvs/author/bin/activate

GALLIUM_DRIVER=d3d12 \
MESA_D3D12_DEFAULT_ADAPTER_NAME=AMD \
python scripts/smoke_test_multi_thor.py \
  --scene FloorPlan1 \
  --thor-executable /home/davei/projects/ai2thor-humanoid/builds/thor-humanoid-Linux64/thor-humanoid-Linux64
```

This checks two bodies, two camera events, and independent movement without involving Qwen.

## 14. Run the v0.8 unit/regression tests

```bash
cd ~/projects/author
source ~/.venvs/author/bin/activate
pytest -q
```

Expected for the released v0.8.0 ZIP:

```text
101 passed
```

Run only the v0.8 texting/manipulation tests:

```bash
pytest -q tests/test_v080.py
```

## 15. Check installed AUTHOR version

```bash
cd ~/projects/author
source ~/.venvs/author/bin/activate
python - <<'PY'
import agent_core
print(agent_core.__version__)
PY
```

Expected:

```text
0.8.0
```

## 16. Fresh v0.8.0 install from ZIP

If `~/projects/author` does not exist:

```bash
cd ~/projects
unzip \
  /mnt/c/Users/davei/Downloads/author_phase9_v0.8.0_social_texting_grounded_manipulation.zip \
  -d ~/projects

cd ~/projects/author
```

Create the venv only if it does not already exist:

```bash
python3 -m venv ~/.venvs/author
source ~/.venvs/author/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest -q
```

If the venv already exists:

```bash
cd ~/projects/author
source ~/.venvs/author/bin/activate
python -m pip install -e ".[dev]"
pytest -q
```

## 17. Upgrade while preserving old training state

Example from v0.7.0:

```bash
cd ~/projects
mv author author_v0.7.0_backup

unzip \
  /mnt/c/Users/davei/Downloads/author_phase9_v0.8.0_social_texting_grounded_manipulation.zip \
  -d ~/projects

cp -a ~/projects/author_v0.7.0_backup/state ~/projects/author/

cd ~/projects/author
source ~/.venvs/author/bin/activate
python -m pip install -e ".[dev]"
pytest -q
```

## 18. LLM server checks and troubleshooting

### WSL — discover Windows host and set endpoint

```bash
WINDOWS_HOST=$(ip route show | awk '/default/ {print $3; exit}')
export AGENT_LLM_BASE_URL="http://$WINDOWS_HOST:8080/v1"
export AGENT_LLM_MODEL="qwen2-vl-2b"
```

### WSL — verify llama.cpp

```bash
curl -s "$AGENT_LLM_BASE_URL/models" | python -m json.tool
```

### Windows PowerShell — verify port 8080 is listening

```powershell
Get-NetTCPConnection -LocalPort 8080 -State Listen
```

### If AUTHOR says `ConnectError: [Errno 111] Connection refused`

1. Start llama-server.exe in Windows.
2. Recompute `WINDOWS_HOST` in the current WSL shell.
3. Re-export `AGENT_LLM_BASE_URL` and `AGENT_LLM_MODEL`.
4. Confirm `/models` with curl before launching AUTHOR.

## 19. Simulation tuning knobs

### `--step-delay`

```text
0       no artificial pause; fastest normal simulation
0.08    previous default
0.2     slower/easier to watch
```

Example:

```bash
--step-delay 0
```

Changing step delay changes wall-clock speed, not the number of LLM observation decisions.

### `--vision-every`

```text
1    camera sent every agent decision; best discovery fidelity
2    every second decision
5    every fifth decision; faster but can miss things
```

For discovery experiments, keep:

```bash
--vision-every 1
```

### `--roam-decisions`

High-level decision/social-round budget.

```bash
--roam-decisions 500
--roam-decisions 2000
```

### `--max-steps`

Primitive Unity action budget per embodied agent.

```bash
--max-steps 3000
--max-steps 10000
```

### `--speech-range`

Override explicit speech hearing radius in meters:

```bash
--speech-range 6.0
```

Texting is distance-independent and controlled separately by `config.yaml`.

### Dashboard refresh

```bash
--dashboard-refresh-ms 350
--dashboard-refresh-ms 1000
```

A slower browser refresh can reduce observer overhead without changing how often the agent itself receives vision.

### Dashboard port

```bash
--dashboard-port 8765
```

### Run normal simulation headless

Optional if the custom player behaves reliably with CloudRendering:

```bash
... python -m agent_core.cli ... --headless ...
```

You can still use the dashboard because AI2-THOR can return rendered frames off-screen.

### Disable vision for debugging only

```bash
--no-vision
```

This changes the agent's experience, so do not use it for discovery training/evaluation.

### Disable object interaction for debugging

```bash
--no-interact
```

## 20. Useful `config.yaml` values

Edit configuration:

```bash
cd ~/projects/author
nano config.yaml
```

Important sections include:

```yaml
simulation:
  step_delay_seconds: 0.08

navigation:
  interaction_distance: 1.25

social:
  speech_range_m: 4.0
  text_enabled: true
  text_max_chars: 800
  max_texts_per_turn: 2

observer:
  port: 8765
  refresh_ms: 350

learning:
  state_dir: state
  load_persistent_memory: true
  memory_max_records: 12000
  replay_max_records: 250000
  use_skill_navigation: true

training:
  episodes: 500
  workers: 1
  image_width: 160
  image_height: 120
  max_steps_per_episode: 120
  peer_encounter_rate: 0.25
  peer_encounter_distance_m: 1.25
```

CLI flags override these values for a single run.

## 21. Environment diagnostics

```bash
cd ~/projects/author
source ~/.venvs/author/bin/activate
python scripts/check_environment.py
```

Check custom player:

```bash
file ~/projects/ai2thor-humanoid/builds/thor-humanoid-Linux64/thor-humanoid-Linux64
ls -lah ~/projects/ai2thor-humanoid/builds/thor-humanoid-Linux64
```

Expected player type includes `ELF 64-bit` and `x86-64`.

## 22. Check humanoid runtime markers

```bash
grep -nEi \
  'v0\.6\.1|humanoid visual replacement|TallVisCap' \
  "$HOME/.config/unity3d/Allen Institute for Artificial Intelligence/AI2-THOR/Player.log"
```

Player log:

```bash
less "$HOME/.config/unity3d/Allen Institute for Artificial Intelligence/AI2-THOR/Player.log"
```

## 23. Find AUTHOR runtime logs

```bash
cd ~/projects/author
find runs -maxdepth 3 -type f -printf '%TY-%Tm-%Td %TH:%TM %p\n' 2>/dev/null | sort
```

Inspect newest files:

```bash
find runs -type f -print0 2>/dev/null | xargs -0 -r ls -lt | head -30
```

## 24. Monitor running simulator processes

```bash
pgrep -af thor-humanoid-Linux64
```

Refresh every two seconds:

```bash
watch -n 2 'pgrep -af thor-humanoid-Linux64'
```

Check CPU/memory interactively:

```bash
top
```

## 25. Stop things cleanly

### Social session

Type:

```text
/quit
```

### llama.cpp

In its Windows PowerShell window press:

```text
Ctrl+C
```

### Stuck WSL Unity player only

First inspect:

```bash
pgrep -af thor-humanoid-Linux64
```

Then, if it really is stale:

```bash
pkill -f thor-humanoid-Linux64
```

## 26. Current persistent-state layout

```text
state/
├── memory/
│   ├── a/memory.jsonl
│   └── b/memory.jsonl
├── maps/
│   └── FloorPlan1.json
├── checkpoints/
│   └── navigation_skill.json
├── replay/
│   └── navigation_replay.jsonl
├── metrics/
│   ├── training_summary.json
│   └── training_history.jsonl
└── messages/
    └── peer_texts.jsonl
```

## 27. Rebuild the humanoid Unity player only when Unity/avatar code changes

v0.8.0 itself does not require this rebuild. These commands are for future Unity/avatar changes.

### WSL — apply/verify humanoid patch

```bash
cd ~/projects/author
source ~/.venvs/author/bin/activate

python unity_humanoid/apply_humanoid_patch.py ~/projects/ai2thor-humanoid
python unity_humanoid/verify_patch.py ~/projects/ai2thor-humanoid
```

### WSL — copy changed C# files into the Windows Unity project

```bash
cp \
  ~/projects/ai2thor-humanoid/unity/Assets/Scripts/AgentManager.cs \
  /mnt/c/Users/davei/a2h/unity-2020.3.48/Assets/Scripts/AgentManager.cs

cp \
  ~/projects/ai2thor-humanoid/unity/Assets/Scripts/HumanoidPeerAvatar.cs \
  /mnt/c/Users/davei/a2h/unity-2020.3.48/Assets/Scripts/HumanoidPeerAvatar.cs
```

### Windows PowerShell — build Linux64 player

```powershell
$Unity48   = "C:\Program Files\Unity\Hub\Editor\2020.3.48f1\Editor\Unity.exe"
$Project48 = "C:\Users\davei\a2h\unity-2020.3.48"
$OutDir    = "C:\Users\davei\a2h\builds\thor-humanoid-Linux64"
$BuildLog  = "C:\Users\davei\a2h\thor-Linux64-humanoid-current.log"

Get-CimInstance Win32_Process -Filter "Name='Unity.exe'" |
  Where-Object { $_.CommandLine -like "*unity-2020.3.48*" } |
  ForEach-Object {
    Stop-Process -Id $_.ProcessId -Force
  }

Start-Sleep -Seconds 2

Remove-Item "$Project48\Temp\UnityLockfile" -Force -ErrorAction SilentlyContinue
Remove-Item $OutDir -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $OutDir | Out-Null
Remove-Item $BuildLog -Force -ErrorAction SilentlyContinue

$env:UNITY_BUILD_NAME = "$OutDir\thor-humanoid-Linux64"
$env:BUILD_SCENES     = "FloorPlan1_physics"

$BuildArgs = @(
  "-quit"
  "-batchmode"
  "-projectPath"
  "`"$Project48`""
  "-buildTarget"
  "Linux64"
  "-executeMethod"
  "Build.Linux64"
  "-logFile"
  "`"$BuildLog`""
)

$BuildProcess = Start-Process `
  -FilePath $Unity48 `
  -ArgumentList $BuildArgs `
  -Wait `
  -PassThru

Write-Host "REAL Unity build exit code: $($BuildProcess.ExitCode)"
Write-Host "Executable:"  (Test-Path "$OutDir\thor-humanoid-Linux64")
Write-Host "Data dir:"    (Test-Path "$OutDir\thor-humanoid-Linux64_Data")
Write-Host "UnityPlayer:" (Test-Path "$OutDir\UnityPlayer.so")
```

Expected:

```text
REAL Unity build exit code: 0
Executable: True
Data dir: True
UnityPlayer: True
```

### WSL — copy rebuilt Linux player back

```bash
rm -rf ~/projects/ai2thor-humanoid/builds/thor-humanoid-Linux64
mkdir -p ~/projects/ai2thor-humanoid/builds/thor-humanoid-Linux64

cp -a \
  /mnt/c/Users/davei/a2h/builds/thor-humanoid-Linux64/. \
  ~/projects/ai2thor-humanoid/builds/thor-humanoid-Linux64/

chmod +x \
  ~/projects/ai2thor-humanoid/builds/thor-humanoid-Linux64/thor-humanoid-Linux64
```

## 28. Recommended routine

For fast skill accumulation:

```bash
cd ~/projects/author
source ~/.venvs/author/bin/activate
EPISODES=5000 WORKERS=2 ./scripts/run_training_wsl.sh --headless
```

Then start llama.cpp and run the normal social session with `--vision-every 1 --step-delay 0`. Do not reset memory between training and normal simulation if you want the learned map/checkpoint/memories to carry over.

## 29. What requires llama.cpp and what does not

```text
Headless accelerated training       NO llama.cpp required
Unity/multi-agent smoke test         NO llama.cpp required
Manipulation smoke test              NO llama.cpp required
pytest                               NO llama.cpp required
Normal autonomous/social session     YES llama.cpp required when --llm is used
Dashboard                            accompanies normal social session
```

## 30. Important safety rules for your data

- `RESET_MEMORY=1` or `--reset-memory` intentionally deletes the configured learning state.
- Back up `state/` before experiments you may want to undo.
- Do not delete the custom AI2-THOR build unless you intend to rebuild/copy it again.
- Run AUTHOR commands from `~/projects/author` so the default `config.yaml` resolves correctly.
- Run PowerShell commands in Windows PowerShell, not in WSL Bash.
- Run Bash commands in WSL, not in Windows PowerShell.
