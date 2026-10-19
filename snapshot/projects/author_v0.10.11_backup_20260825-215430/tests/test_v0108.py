from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _builder() -> str:
    return (ROOT / "unity_world" / "DiscoveryWorldBuilder.cs").read_text(encoding="utf-8")


def test_release_version_is_v0108():
    import agent_core
    assert agent_core.__version__ == "0.10.11"


def test_discovery_world_removes_stools_before_navmesh_build():
    text = _builder()
    assert "RemoveStoolsFromScene(scene)" in text
    assert "ValidateNoStoolsRemain(scene)" in text
    assert 'IndexOf("stool", StringComparison.OrdinalIgnoreCase)' in text
    assert text.index("RemoveStoolsFromScene(scene)") < text.index("NavMeshBuilder.BuildNavMesh()")


def test_generated_rooms_have_explicit_turnaround_and_door_corridor_contracts():
    text = _builder()
    assert "TurnaroundSize = 2.20f" in text
    assert "DoorCorridorWidth = 1.25f" in text
    assert 'ValidateRoomManeuvering(root.transform, "NorthArchive"' in text
    assert 'ValidateRoomManeuvering(root.transform, "EastExperiment"' in text
    assert "northTurn" in text and "northCorridor" in text
    assert "eastTurn" in text and "eastCorridor" in text


def test_new_room_walls_have_keepout_buffers_to_prevent_wall_hugging_starts():
    text = _builder()
    for token in (
        "NorthManeuverBuffer_West",
        "NorthManeuverBuffer_East",
        "NorthManeuverBuffer_Far",
        "EastManeuverBuffer_Far",
        "EastManeuverBuffer_North",
        "EastManeuverBuffer_South",
    ):
        assert token in text
    assert "WallKeepOutDepth = 0.35f" in text


def test_generated_rooms_no_longer_place_loose_floor_box_or_statue():
    text = _builder()
    assert "PlaceStockObject(SimObjType.Statue" not in text
    assert "PlaceStockObject(SimObjType.Box" not in text


def test_furniture_is_moved_out_of_reserved_room_centers():
    text = _builder()
    assert 'CreateStaticCube("ArchiveDesk", new Vector3(-1.20f, 0.42f, 7.00f)' in text
    assert 'CreateStaticCube("ArchivePedestal", new Vector3(1.15f, 0.52f, 7.00f)' in text
    assert 'CreateStaticCube("ExperimentBench", new Vector3(6.90f, 0.42f, -0.40f), new Vector3(0.80f, 0.84f, 2.25f)' in text
    assert 'CreateStaticCube("ArtifactPedestalA", new Vector3(4.45f, 0.5f, 2.10f)' in text
    assert 'CreateStaticCube("ArtifactPedestalB", new Vector3(6.20f, 0.5f, 2.10f)' in text


def test_runtime_smoke_test_checks_stools_and_maneuver_cells():
    text = (ROOT / "scripts" / "smoke_test_discovery_world.py").read_text(encoding="utf-8")
    assert "Stool objects:" in text
    assert "north_turn" in text and "east_turn" in text
    assert "north_corridor" in text and "east_corridor" in text


def test_windows_sync_also_updates_discovery_build_script():
    text = (ROOT / "scripts" / "sync_discovery_unity_to_windows.sh").read_text(encoding="utf-8")
    assert 'WIN_TOOLS_DIR="${WIN_TOOLS_DIR:-/mnt/c/Users/davei/a2h}"' in text
    assert 'cp "scripts/build_discovery_world.ps1" "$WIN_TOOLS_DIR/build_discovery_world.ps1"' in text


def test_post_build_sync_moves_windows_player_into_author_wsl_path():
    text = (ROOT / "scripts" / "sync_discovery_build_to_wsl.sh").read_text(encoding="utf-8")
    assert '/mnt/c/Users/davei/a2h/builds/thor-discovery-Linux64' in text
    assert '$HOME/projects/ai2thor-humanoid/builds/thor-discovery-Linux64' in text
    assert 'chmod +x "$WSL_BUILD/thor-discovery-Linux64"' in text
