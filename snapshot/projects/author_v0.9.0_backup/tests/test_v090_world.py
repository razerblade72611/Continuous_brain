from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def test_v090_builder_contains_two_rooms_and_native_openable_doors():
    text=(ROOT/'unity_world/DiscoveryWorldBuilder.cs').read_text()
    assert 'NorthArchiveRoom' in text
    assert 'EastExperimentRoom' in text
    assert 'SimObjType.Door' in text
    assert 'SimObjSecondaryProperty.CanOpen' in text
    assert 'CanOpen_Object' in text
    assert 'NavMeshBuilder.BuildNavMesh' in text
    assert 'FloorPlan1Discovery_physics.unity' in text

def test_v090_patcher_is_non_destructive_to_original_scene():
    text=(ROOT/'unity_world/apply_discovery_world_patch.py').read_text()
    assert 'FloorPlan1_physics.unity' in text
    assert 'Door = 162' in text
    builder=(ROOT/'unity_world/DiscoveryWorldBuilder.cs').read_text()
    assert 'AssetDatabase.CopyAsset(SourceScene, OutputScene)' in builder

def test_discovery_training_helper_defaults_to_custom_scene():
    text=(ROOT/'scripts/run_training_wsl.sh').read_text()
    assert 'SCENE="${SCENE:-FloorPlan1Discovery_physics}"' in text
    assert '--scene "$SCENE"' in text

def test_discovery_session_helper_uses_custom_player_and_scene():
    text=(ROOT/'scripts/run_discovery_session_wsl.sh').read_text()
    assert 'FloorPlan1Discovery_physics' in text
    assert 'thor-discovery-Linux64' in text

def test_text_tasks_require_delivery_not_claim_only():
    text=(ROOT/'src/agent_core/social_session.py').read_text()
    assert 'text_verified' in text
    assert '_task_text_deliveries' in text
    assert 'claimed and not structured and not text_task' in text
