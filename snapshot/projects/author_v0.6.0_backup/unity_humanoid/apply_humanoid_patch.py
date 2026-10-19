#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

MARKER = "// V060_HUMANOID_SOCIAL_EMBODIMENT"
HOOK = '''        // V060_HUMANOID_SOCIAL_EMBODIMENT
        // Replace only peer-visible renderers. CharacterController/camera/physics remain stock.
        if (action.makeAgentsVisible) {
            HumanoidPeerAvatar.ApplyAll(this.agents, this.agentColors);
        }
'''


def _patch_agent_manager(text: str) -> tuple[str, str]:
    """Patch either AI2-THOR 4.3 coroutine layout or newer synchronous layout."""
    if MARKER in text:
        return text, "already-patched"

    # AI2-THOR 4.3.0 (commit a634e28): Initialize starts StartCoroutine(addAgents(action));
    # and the actual agent list is complete inside private IEnumerator addAgents(...).
    method_sig = "private IEnumerator addAgents(ServerAction action)"
    if method_sig in text:
        method_start = text.index(method_sig)
        # In 4.3, this loop immediately follows creation of all additional bodies.
        anchor = "        for (int i = 0; i < this.agents.Count; i++) {\n            this.agents[i].m_Camera.depth = 1;\n        }"
        anchor_pos = text.find(anchor, method_start)
        if anchor_pos == -1:
            raise RuntimeError(
                "Found AI2-THOR 4.3-style addAgents coroutine but not the expected camera-depth anchor. "
                "Refusing to guess a patch location."
            )
        return text[:anchor_pos] + HOOK + text[anchor_pos:], "ai2thor-4.3-coroutine"

    # Newer AI2-THOR source currently performs addAgents(action) synchronously.
    sync_anchor = "        addAgents(action);"
    if sync_anchor in text:
        replacement = sync_anchor + "\n\n" + HOOK.rstrip("\n")
        return text.replace(sync_anchor, replacement, 1), "modern-synchronous"

    raise RuntimeError(
        "Could not find a supported AI2-THOR multi-agent initialization layout. "
        "Expected either the 4.3 addAgents coroutine or modern addAgents(action); call."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Patch an AI2-THOR source checkout with the v0.6 humanoid peer renderer.")
    parser.add_argument("ai2thor_root", help="Path to the cloned allenai/ai2thor repository")
    parser.add_argument("--revert", action="store_true", help="Restore AgentManager.cs from the .v060.bak backup and remove HumanoidPeerAvatar.cs")
    args = parser.parse_args()

    root = Path(args.ai2thor_root).expanduser().resolve()
    scripts = root / "unity" / "Assets" / "Scripts"
    manager = scripts / "AgentManager.cs"
    avatar_dst = scripts / "HumanoidPeerAvatar.cs"
    backup = manager.with_suffix(manager.suffix + ".v060.bak")
    avatar_src = Path(__file__).with_name("HumanoidPeerAvatar.cs")

    if not manager.exists():
        raise SystemExit(f"AgentManager.cs not found: {manager}")

    if args.revert:
        if not backup.exists():
            raise SystemExit(f"Backup not found: {backup}")
        shutil.copy2(backup, manager)
        if avatar_dst.exists():
            avatar_dst.unlink()
        print("v0.6 humanoid patch reverted.")
        return 0

    raw = manager.read_text(encoding="utf-8-sig")
    try:
        patched, layout = _patch_agent_manager(raw)
    except RuntimeError as exc:
        raise SystemExit(str(exc))

    if MARKER not in raw and not backup.exists():
        shutil.copy2(manager, backup)
    manager.write_text(patched, encoding="utf-8")
    shutil.copy2(avatar_src, avatar_dst)

    final = manager.read_text(encoding="utf-8")
    checks = {
        "AgentManager marker": MARKER in final,
        "ApplyAll call": "HumanoidPeerAvatar.ApplyAll" in final,
        "avatar source": avatar_dst.exists(),
        "backup": backup.exists(),
    }
    for name, ok in checks.items():
        print(f"{name}: {'OK' if ok else 'FAILED'}")
    if not all(checks.values()):
        return 2
    print(f"Patch applied using layout: {layout}")
    print("Open the Unity project or run an AI2-THOR local build next.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
