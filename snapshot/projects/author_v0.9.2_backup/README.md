# AI2-THOR Agent v0.9.2 — Walkable Discovery + Visible Peer Link

v0.9.2 fixes the physical reachability of both generated Discovery Rooms and adds a reliable, visible A↔B peer-text path.

## v0.9.2 highlights

- Walkable room and threshold collider children are named exactly `Floor`, matching AI2-THOR 4.3.0 `GetReachablePositions` collision semantics.
- Both native `Door` objects and the separate `FloorPlan1Discovery_physics` scene are retained.
- Explicit peer-text tasks recover with a dedicated LLM message-composition call if the normal embodied action JSON fails to emit `TextPeer`.
- Periodic peer text is configurable with `--peer-text-every N`; the shipped social config defaults to 3 turns.
- Delivered messages print in the terminal, appear in the dashboard event stream, persist to `state/messages/peer_texts.jsonl`, and enter the recipient LLM's next turn.
- A visibly labeled `[status relay]` fallback can preserve transport if the peer-message composition call fails; it uses only sender-public task/observation information.
- v0.9.1 verified task completion, grounded manipulation handoff, text delivery verification, physical Discovery-region escape verification, and the Unity SceneManager compiler fix are retained.

## Install / upgrade

```bash
cd ~/projects
mv author author_v0.9.1_backup
unzip /mnt/c/Users/davei/Downloads/author_phase9_v0.9.2_walkable_discovery_visible_peer_link.zip -d ~/projects
cp -a ~/projects/author_v0.9.1_backup/state ~/projects/author/
cd ~/projects/author
source ~/.venvs/author/bin/activate
python -m pip install -e ".[dev]"
pytest -q
```

See `AUTHOR_v0.9.2_OPERATOR_README.md` for Unity rebuild, smoke-test, launch, and peer-text commands.
