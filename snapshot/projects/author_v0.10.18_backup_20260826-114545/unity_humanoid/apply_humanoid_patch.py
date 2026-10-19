#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

MARKER = "// V061_HUMANOID_VISUAL_REPLACEMENT"
OLD_MARKER = "// V060_HUMANOID_SOCIAL_EMBODIMENT"
HOOK = '''        // V061_HUMANOID_VISUAL_REPLACEMENT
        // Visual-only replacement inside AI2-THOR's native TallVisCap hierarchy.
        if (action.makeAgentsVisible) {
            HumanoidPeerAvatar.ApplyAll(this.agents, this.agentColors);
        }
'''


def _upgrade_v060_hook(text: str) -> tuple[str, str]:
    if OLD_MARKER not in text:
        return text, "not-v060"
    upgraded = text.replace(OLD_MARKER, MARKER, 1)
    upgraded = upgraded.replace(
        "// Replace only peer-visible renderers. CharacterController/camera/physics remain stock.",
        "// Visual-only replacement inside AI2-THOR's native TallVisCap hierarchy.",
        1,
    )
    return upgraded, "upgrade-v0.6.0"


def _patch_agent_manager(text: str) -> tuple[str, str]:
    """Patch AI2-THOR 4.3 or newer layouts, including an in-place v0.6.0 upgrade."""
    if MARKER in text:
        return text, "already-patched"

    if OLD_MARKER in text:
        return _upgrade_v060_hook(text)

    method_sig = "private IEnumerator addAgents(ServerAction action)"
    if method_sig in text:
        method_start = text.index(method_sig)
        anchor = "        for (int i = 0; i < this.agents.Count; i++) {\n            this.agents[i].m_Camera.depth = 1;\n        }"
        anchor_pos = text.find(anchor, method_start)
        if anchor_pos == -1:
            raise RuntimeError(
                "Found AI2-THOR 4.3-style addAgents coroutine but not the expected camera-depth anchor. "
                "Refusing to guess a patch location."
            )
        return text[:anchor_pos] + HOOK + text[anchor_pos:], "ai2thor-4.3-coroutine"

    sync_anchor = "        addAgents(action);"
    if sync_anchor in text:
        replacement = sync_anchor + "\n\n" + HOOK.rstrip("\n")
        return text.replace(sync_anchor, replacement, 1), "modern-synchronous"

    raise RuntimeError(
        "Could not find a supported AI2-THOR multi-agent initialization layout. "
        "Expected either the 4.3 addAgents coroutine or modern addAgents(action); call."
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Patch an AI2-THOR source checkout with the v0.6.1 TallVisCap humanoid visual replacement."
    )
    parser.add_argument("ai2thor_root", help="Path to the cloned allenai/ai2thor repository")
    parser.add_argument(
        "--revert",
        action="store_true",
        help="Restore the exact pre-v0.6.1 AgentManager/avatar state from .v061 backups",
    )
    args = parser.parse_args()

    root = Path(args.ai2thor_root).expanduser().resolve()
    scripts = root / "unity" / "Assets" / "Scripts"
    manager = scripts / "AgentManager.cs"
    avatar_dst = scripts / "HumanoidPeerAvatar.cs"
    manager_backup = manager.with_suffix(manager.suffix + ".v061.bak")
    avatar_backup = scripts / "HumanoidPeerAvatar.cs.v061.bak"
    avatar_src = Path(__file__).with_name("HumanoidPeerAvatar.cs")

    if not manager.exists():
        raise SystemExit(f"AgentManager.cs not found: {manager}")

    if args.revert:
        if not manager_backup.exists():
            raise SystemExit(f"Backup not found: {manager_backup}")
        shutil.copy2(manager_backup, manager)
        if avatar_backup.exists():
            shutil.copy2(avatar_backup, avatar_dst)
        elif avatar_dst.exists():
            avatar_dst.unlink()
        print("v0.6.1 humanoid visual replacement reverted to the exact pre-v0.6.1 state.")
        return 0

    raw = manager.read_text(encoding="utf-8-sig")
    try:
        patched, layout = _patch_agent_manager(raw)
    except RuntimeError as exc:
        raise SystemExit(str(exc))

    # Back up the user's current state once. On a v0.6.0 checkout this means
    # --revert returns to v0.6.0 rather than all the way to pristine AI2-THOR.
    if not manager_backup.exists():
        shutil.copy2(manager, manager_backup)
    if avatar_dst.exists() and not avatar_backup.exists():
        shutil.copy2(avatar_dst, avatar_backup)

    manager.write_text(patched, encoding="utf-8")
    shutil.copy2(avatar_src, avatar_dst)

    final = manager.read_text(encoding="utf-8-sig")
    avatar_text = avatar_dst.read_text(encoding="utf-8-sig")
    checks = {
        "AgentManager v0.6.1 marker": MARKER in final,
        "v0.6.0 marker removed": OLD_MARKER not in final,
        "ApplyAll call": "HumanoidPeerAvatar.ApplyAll" in final,
        "avatar source": avatar_dst.exists(),
        "TallVisCap host": "agent.TallVisCap" in avatar_text,
        "stock force-render suppression": "forceRenderingOff = true" in avatar_text,
        "v0.6.1 runtime marker": "v0.6.1 humanoid visual replacement applied" in avatar_text,
        "manager backup": manager_backup.exists(),
    }
    for name, ok in checks.items():
        print(f"{name}: {'OK' if ok else 'FAILED'}")
    if not all(checks.values()):
        return 2

    print(f"Patch applied using layout: {layout}")
    print("Rebuild the custom Unity player next; the Python/social runtime does not need behavioral changes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
