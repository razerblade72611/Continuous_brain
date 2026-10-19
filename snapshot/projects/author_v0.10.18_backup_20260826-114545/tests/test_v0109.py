from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_release_version_is_v0109():
    import agent_core
    assert agent_core.__version__ == "0.10.18"


def test_runtime_sanitizer_is_shipped_and_attached():
    sanitizer = (ROOT / "unity_world" / "DiscoveryRuntimeSanitizer.cs").read_text(encoding="utf-8")
    builder = (ROOT / "unity_world" / "DiscoveryWorldBuilder.cs").read_text(encoding="utf-8")
    assert "DefaultExecutionOrder(-32000)" in sanitizer
    assert "Resources.FindObjectsOfTypeAll<SimObjPhysics>()" in sanitizer
    assert 'IndexOf("stool", StringComparison.OrdinalIgnoreCase)' in sanitizer
    assert "DestroyImmediate(go)" in sanitizer
    assert "AddComponent<DiscoveryRuntimeSanitizer>()" in builder


def test_editor_stool_removal_also_scans_runtime_registry_view():
    builder = (ROOT / "unity_world" / "DiscoveryWorldBuilder.cs").read_text(encoding="utf-8")
    assert "Resources.FindObjectsOfTypeAll<SimObjPhysics>()" in builder
    assert "go.scene != scene" in builder


def test_patch_installer_and_windows_sync_copy_runtime_sanitizer():
    apply_text=(ROOT / "unity_world" / "apply_discovery_world_patch.py").read_text(encoding="utf-8")
    sync_text=(ROOT / "scripts" / "sync_discovery_unity_to_windows.sh").read_text(encoding="utf-8")
    verify_text=(ROOT / "unity_world" / "verify_discovery_world_patch.py").read_text(encoding="utf-8")
    assert 'DiscoveryRuntimeSanitizer.cs' in apply_text
    assert 'DiscoveryRuntimeSanitizer.cs' in sync_text
    assert 'DiscoveryRuntimeSanitizer.cs' in verify_text
