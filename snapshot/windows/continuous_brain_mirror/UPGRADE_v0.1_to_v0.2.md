# Upgrade Continuous Brain v0.1 -> v0.2

v0.2 is checkpoint-compatible with v0.1. Preserve the entire `state/` directory and the learned brain resumes with the new visualizer.

## 1. Stop v0.1 cleanly

Press `Ctrl+C` and wait for the prompt to return. This saves `brain.pt`, `status.json`, and closes `episodic.sqlite3`.

## 2. From Ubuntu-24.04, back up v0.1

```bash
cd ~/projects
mv continuous_brain_project continuous_brain_v0.1_backup
```

## 3. Extract v0.2

Assuming the ZIP is in Windows Downloads:

```bash
unzip /mnt/c/Users/davei/Downloads/continuous_brain_v0.2.0.zip -d ~/projects
cd ~/projects/continuous_brain_project
```

## 4. Create a fresh venv

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
```

For the known-safe CPU configuration:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install "numpy>=1.24"
python -m pip install -e . --no-deps
python -m pip install pytest
```

## 5. Restore the learned state

```bash
cp -a ~/projects/continuous_brain_v0.1_backup/state ./
```

Confirm:

```bash
find state/my_brain -maxdepth 1 -type f -ls
```

You should see at least `brain.pt`, `episodic.sqlite3`, and `status.json` if they existed in v0.1.

## 6. Test

```bash
pytest -q
```

## 7. Launch visually

```bash
continuous-brain --state-dir state/my_brain --visual
```

Open:

```text
http://localhost:8765
```

Or:

```bash
continuous-brain --state-dir state/my_brain --visual --open-browser
```

The step/reward/goal counters loaded from v0.1 should continue rather than resetting to zero.
