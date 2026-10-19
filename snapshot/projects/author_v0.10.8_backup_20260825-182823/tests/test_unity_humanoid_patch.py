from pathlib import Path
import subprocess
import sys


def _patcher() -> Path:
    return Path(__file__).parents[1] / "unity_humanoid" / "apply_humanoid_patch.py"


def _make_root(tmp_path: Path, name: str = "ai2thor") -> tuple[Path, Path]:
    root = tmp_path / name
    scripts = root / "unity" / "Assets" / "Scripts"
    scripts.mkdir(parents=True)
    return root, scripts


def test_humanoid_patch_script_inserts_hook_and_avatar(tmp_path):
    root, scripts = _make_root(tmp_path)
    manager = scripts / "AgentManager.cs"
    manager.write_text(
        "public class AgentManager {\n    void Initialize() {\n        addAgents(action);\n    }\n}\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, str(_patcher()), str(root)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    patched = manager.read_text(encoding="utf-8")
    assert "V061_HUMANOID_VISUAL_REPLACEMENT" in patched
    assert "V060_HUMANOID_SOCIAL_EMBODIMENT" not in patched
    assert "HumanoidPeerAvatar.ApplyAll" in patched
    assert (scripts / "HumanoidPeerAvatar.cs").exists()
    assert (scripts / "AgentManager.cs.v061.bak").exists()


def test_humanoid_patch_revert_restores_original(tmp_path):
    root, scripts = _make_root(tmp_path)
    manager = scripts / "AgentManager.cs"
    original = "public class AgentManager {\n    void Initialize() {\n        addAgents(action);\n    }\n}\n"
    manager.write_text(original, encoding="utf-8")
    subprocess.check_call([sys.executable, str(_patcher()), str(root)])
    subprocess.check_call([sys.executable, str(_patcher()), str(root), "--revert"])
    assert manager.read_text(encoding="utf-8") == original
    assert not (scripts / "HumanoidPeerAvatar.cs").exists()


def test_humanoid_patch_supports_ai2thor_430_coroutine_layout(tmp_path):
    root, scripts = _make_root(tmp_path, "ai2thor430")
    manager = scripts / "AgentManager.cs"
    manager.write_text(
        '''using System.Collections;\npublic class AgentManager {\n'''
        '''private IEnumerator addAgents(ServerAction action) {\n'''
        '''    yield return null;\n'''
        '''    for (int i = 1; i < action.agentCount; i++) { addAgent(action); yield return null; }\n'''
        '''        for (int i = 0; i < this.agents.Count; i++) {\n'''
        '''            this.agents[i].m_Camera.depth = 1;\n'''
        '''        }\n'''
        '''}\n}\n''',
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, str(_patcher()), str(root)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    patched = manager.read_text(encoding="utf-8")
    assert "Patch applied using layout: ai2thor-4.3-coroutine" in result.stdout
    assert patched.index("HumanoidPeerAvatar.ApplyAll") < patched.index("this.agents[i].m_Camera.depth = 1")


def test_v060_checkout_is_upgraded_in_place_and_revert_returns_to_v060(tmp_path):
    root, scripts = _make_root(tmp_path, "ai2thor-v060")
    manager = scripts / "AgentManager.cs"
    v060_manager = '''public class AgentManager {\nprivate IEnumerator addAgents(ServerAction action) {\n'''
    v060_manager += '''        // V060_HUMANOID_SOCIAL_EMBODIMENT\n'''
    v060_manager += '''        // Replace only peer-visible renderers. CharacterController/camera/physics remain stock.\n'''
    v060_manager += '''        if (action.makeAgentsVisible) {\n            HumanoidPeerAvatar.ApplyAll(this.agents, this.agentColors);\n        }\n'''
    v060_manager += '''        for (int i = 0; i < this.agents.Count; i++) {\n            this.agents[i].m_Camera.depth = 1;\n        }\n}\n}\n'''
    manager.write_text(v060_manager, encoding="utf-8")
    old_avatar = scripts / "HumanoidPeerAvatar.cs"
    old_avatar.write_text("// v0.6 old avatar\n", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(_patcher()), str(root)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    upgraded = manager.read_text(encoding="utf-8")
    assert "V061_HUMANOID_VISUAL_REPLACEMENT" in upgraded
    assert "V060_HUMANOID_SOCIAL_EMBODIMENT" not in upgraded
    assert upgraded.count("HumanoidPeerAvatar.ApplyAll") == 1
    assert "Patch applied using layout: upgrade-v0.6.0" in result.stdout
    assert (scripts / "HumanoidPeerAvatar.cs.v061.bak").read_text(encoding="utf-8") == "// v0.6 old avatar\n"

    subprocess.check_call([sys.executable, str(_patcher()), str(root), "--revert"])
    assert manager.read_text(encoding="utf-8") == v060_manager
    assert old_avatar.read_text(encoding="utf-8") == "// v0.6 old avatar\n"


def test_v061_avatar_uses_native_tallviscap_and_persistent_stock_suppression():
    avatar = (Path(__file__).parents[1] / "unity_humanoid" / "HumanoidPeerAvatar.cs").read_text(encoding="utf-8")
    assert 'RootName = "V061HumanoidPeerAvatar"' in avatar
    assert "agent.TallVisCap" in avatar
    assert "agent.VisibilityCapsule = host" in avatar
    assert "renderer.forceRenderingOff = true" in avatar
    assert "renderer.shadowCastingMode = ShadowCastingMode.Off" in avatar
    assert "renderer.forceRenderingOff = false" in avatar
    assert "UpdateCull(agent, host)" in avatar
    assert "agent.IsVisible = true" not in avatar


def test_v061_avatar_coordinates_are_centered_on_ai2thor_tall_body():
    avatar = (Path(__file__).parents[1] / "unity_humanoid" / "HumanoidPeerAvatar.cs").read_text(encoding="utf-8")
    # Regression against v0.6.0, which treated the centered agent transform as floor level.
    assert '"Head", new Vector3(0f, 0.70f' in avatar
    assert '"FootL", new Vector3(-0.14f, -0.86f' in avatar
    assert "root.position = agentTransform.position" in avatar
    assert "SafeInverse(hostScale.y)" in avatar
