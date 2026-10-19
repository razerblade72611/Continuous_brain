from pathlib import Path
import subprocess
import sys


def test_humanoid_patch_script_inserts_hook_and_avatar(tmp_path):
    root = tmp_path / "ai2thor"
    scripts = root / "unity" / "Assets" / "Scripts"
    scripts.mkdir(parents=True)
    manager = scripts / "AgentManager.cs"
    manager.write_text(
        "public class AgentManager {\n    void Initialize() {\n        addAgents(action);\n    }\n}\n",
        encoding="utf-8",
    )
    patcher = Path(__file__).parents[1] / "unity_humanoid" / "apply_humanoid_patch.py"
    result = subprocess.run(
        [sys.executable, str(patcher), str(root)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    patched = manager.read_text(encoding="utf-8")
    assert "V060_HUMANOID_SOCIAL_EMBODIMENT" in patched
    assert "HumanoidPeerAvatar.ApplyAll" in patched
    assert (scripts / "HumanoidPeerAvatar.cs").exists()
    assert (scripts / "AgentManager.cs.v060.bak").exists()


def test_humanoid_patch_revert_restores_original(tmp_path):
    root = tmp_path / "ai2thor"
    scripts = root / "unity" / "Assets" / "Scripts"
    scripts.mkdir(parents=True)
    manager = scripts / "AgentManager.cs"
    original = "public class AgentManager {\n    void Initialize() {\n        addAgents(action);\n    }\n}\n"
    manager.write_text(original, encoding="utf-8")
    patcher = Path(__file__).parents[1] / "unity_humanoid" / "apply_humanoid_patch.py"
    subprocess.check_call([sys.executable, str(patcher), str(root)])
    subprocess.check_call([sys.executable, str(patcher), str(root), "--revert"])
    assert manager.read_text(encoding="utf-8") == original
    assert not (scripts / "HumanoidPeerAvatar.cs").exists()


def test_humanoid_patch_supports_ai2thor_430_coroutine_layout(tmp_path):
    root = tmp_path / "ai2thor430"
    scripts = root / "unity" / "Assets" / "Scripts"
    scripts.mkdir(parents=True)
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
    patcher = Path(__file__).parents[1] / "unity_humanoid" / "apply_humanoid_patch.py"
    result = subprocess.run([sys.executable, str(patcher), str(root)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    patched = manager.read_text(encoding="utf-8")
    assert "Patch applied using layout: ai2thor-4.3-coroutine" in result.stdout
    assert patched.index("HumanoidPeerAvatar.ApplyAll") < patched.index("this.agents[i].m_Camera.depth = 1")
