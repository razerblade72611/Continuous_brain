# AUTHOR v0.10.8 Operator Guide — Maneuver-Clear Discovery Rooms

## Why this release exists

During Level-3 visual inspection, one humanoid became trapped on a tipped stool and the other could enter a corner/pocket in a generated Discovery room. These are environment geometry defects, not skill failures.

v0.10.8 keeps the v0.10.7 selected-event fix and v0.10.6 simulator watchdog, while updating the Unity Discovery World builder from v0.9.3 to v0.9.4.

## Geometry contract

The North Archive and East Experiment remain 5 m x 5 m rooms. Each now reserves:

- a 2.20 m x 2.20 m collider-free turnaround square at room center;
- a 1.25 m-wide collider-free corridor from the doorway to that turnaround square;
- a 0.35 m invisible keep-out strip along the three non-door walls.

The default AI2-THOR physics capsule radius is 0.20 m, so the wall keep-out plus capsule radius keeps the agent center roughly 0.55 m away from the visible wall. This is intentionally much larger than bare collision clearance.

Furniture is moved to the perimeter. Loose generated-room floor `Statue` and `Box` props are removed. Narrow pockets behind the desk, bench, and pedestals are intentionally sealed rather than left as reachable dead space.

All stools in the cloned FloorPlan1 scene are removed before NavMesh generation. The build fails if a stool-like SimObj or named GameObject remains.

## North Archive layout

```text
              FAR WALL / SCREEN
+------------------------------------------------+
| keep-out                                       |
|   DESK                 PEDESTAL                |
|                                                |
|          +----------------------+              |
|          |   2.2 m TURN ZONE   |              |
|          |                      |              |
|          +----------+-----------+              |
|                     |                          |
|              1.25 m | corridor                 |
+---------------------DOOR-----------------------+
```

## East Experiment layout

```text
+---------------- NORTH WALL --------------------+
| keep-out     PEDESTAL       PEDESTAL           |
|                                                |
|        +----------------------+        BENCH   |
| DOOR---|   2.2 m TURN ZONE   |-------- BENCH   |
|        +----------------------+        BENCH   |
|                                                |
| keep-out                                       |
+---------------- SOUTH WALL --------------------+
                               FAR WALL / SCREEN
```

## Important: rebuilding AUTHOR alone is not enough

The room changes live in `unity_world/DiscoveryWorldBuilder.cs`, so the custom Unity player must be regenerated and rebuilt.

### 1. Upgrade AUTHOR while preserving A/B state

```bash
cd ~/projects
mv author author_v0.10.7_backup
unzip /mnt/c/Users/davei/Downloads/author_phase9_v0.10.8_maneuver_clear_rooms.zip -d ~/projects
cp -a ~/projects/author_v0.10.7_backup/state ~/projects/author/

cd ~/projects/author
source ~/.venvs/author/bin/activate
python -m pip install -e ".[dev]"
pytest -q
```

Expected: `179 passed`.

Never use `--reset-memory` for this upgrade.

### 2. Install the v0.9.4 Discovery builder into the WSL AI2-THOR checkout

```bash
cd ~/projects/author
source ~/.venvs/author/bin/activate

python unity_world/apply_discovery_world_patch.py \
  ~/projects/ai2thor-humanoid

python unity_world/verify_discovery_world_patch.py \
  ~/projects/ai2thor-humanoid
```

### 3. Sync patched Unity source into the Windows Unity project

```bash
cd ~/projects/author
./scripts/sync_discovery_unity_to_windows.sh
```

Default Windows target:

```text
C:\Users\davei\a2h\unity-2020.3.48
```

### 4. Regenerate the Discovery scene and rebuild the Linux player

Run from Windows PowerShell:

```powershell
& C:\Users\davei\a2h\build_discovery_world.ps1
```

Expected final lines include successful scene generation/build and an existing executable.

### 5. Sync the rebuilt Windows player back into WSL

AUTHOR launches the WSL copy, so do not skip this step or you may accidentally test the old geometry:

```bash
cd ~/projects/author
./scripts/sync_discovery_build_to_wsl.sh
```

It copies from:

```text
/mnt/c/Users/davei/a2h/builds/thor-discovery-Linux64
```

to:

```text
/home/davei/projects/ai2thor-humanoid/builds/thor-discovery-Linux64
```

Then run the geometry smoke test:

```bash
cd ~/projects/author
source ~/.venvs/author/bin/activate

python scripts/smoke_test_discovery_world.py \
  --thor-executable /home/davei/projects/ai2thor-humanoid/builds/thor-discovery-Linux64/thor-discovery-Linux64
```

The smoke test now verifies:

- two openable discovery doors;
- both room extensions reachable;
- no Stool objects remain;
- both 2.2 m turnaround regions contain reachable cells;
- both 1.25 m doorway corridors contain reachable cells;
- interactive room content remains available.

## Training validation

After the rebuilt Unity world passes the smoke test, use a short Level-3 run first:

```bash
EPISODES=60 WORKERS=1 \
./scripts/run_training_wsl.sh \
  --curriculum door \
  --level 3
```

The new training summary identifies version `0.10.8` and trainer `maneuver_clear_discovery_curriculum`.

If the 60-episode run is clean, continue to the full Level-3 qualification.
