from __future__ import annotations

import json
from pathlib import Path

from agent_core.embodied_skills import EmbodiedSkillLibrary
from agent_core.interaction_pose import InteractionPose, choose_best_interaction_pose
from agent_core.navigation import NavigationConfig
from agent_core.social import TextBus, normalize_peer_message_text
from agent_core.training import _curriculum_from_config, _navigation_pair_for_level, build_parser


def test_peer_transport_rejects_sentinels_and_malformed_json(tmp_path: Path):
    log = tmp_path / "peer_texts.jsonl"
    bus = TextBus({0: "A", 1: "B"}, log_path=log)
    for bad in ("NO_MESSAGE", "[NO_MESSAGE]", "null", '{"text":', '{"analysis":"private"}'):
        assert bus.send(0, "B", bad) is None
    assert bus.peek(1) == []
    assert not log.exists() or not log.read_text().strip()


def test_peer_transport_persists_unique_id_only_after_delivery(tmp_path: Path):
    log = tmp_path / "peer_texts.jsonl"
    bus = TextBus({0: "A", 1: "B"}, log_path=log)
    event = bus.send(0, "B", '{"text":"Meet me at the door."}')
    assert event is not None
    assert len(event.message_id) == 32
    assert bus.peek(1)[0].message_id == event.message_id
    row = json.loads(log.read_text().strip())
    assert row["message_id"] == event.message_id
    assert row["text"] == "Meet me at the door."
    assert event.for_llm() == {"from": "A", "text": "Meet me at the door.", "channel": "text"}


def test_peer_normalizer_fails_closed_on_structured_garbage():
    assert normalize_peer_message_text("```json\n{bad}\n```") == ""
    assert normalize_peer_message_text('["hello"]') == "hello"
    assert normalize_peer_message_text('["a", "b"]') == ""


def test_v010_checkpoint_migrates_losslessly_and_does_not_trust_legacy_navigate(tmp_path: Path):
    path = tmp_path / "skills.json"
    old = {
        "version": 1,
        "agent_label": "A",
        "skills": {
            "navigate": {"attempts": 100, "successes": 99, "failures": 1, "primitive_steps": 200, "recoveries": 0},
            "open": {"attempts": 10, "successes": 10, "failures": 0, "primitive_steps": 10, "recoveries": 0},
        },
        "by_object_type": {},
    }
    path.write_text(json.dumps(old))
    lib = EmbodiedSkillLibrary(path, agent_label="A")
    assert lib.ready("door_open", "Door", min_attempts=5, min_success_rate=0.9)
    assert not lib.ready("door_approach", "Door", min_attempts=5, min_success_rate=0.1)
    lib.save()
    saved = json.loads(path.read_text())
    assert saved["version"] == 3
    assert saved["legacy_v010"]["skills"] == old["skills"]


def test_legacy_skill_writes_land_in_staged_buckets(tmp_path: Path):
    lib = EmbodiedSkillLibrary(tmp_path / "skills.json", agent_label="A")
    lib.record("cross_door", success=True, primitive_steps=4, object_type="Door")
    assert lib.stats("door_cross", "Door").successes == 1
    assert lib.stats("cross_door", "Door").successes == 1


def test_interaction_pose_ranking_uses_connected_grid_path():
    reachable = [
        {"x": 0.0, "y": 0.9, "z": 0.0},
        {"x": 0.25, "y": 0.9, "z": 0.0},
        {"x": 0.5, "y": 0.9, "z": 0.0},
        {"x": 0.75, "y": 0.9, "z": 0.0},
    ]
    poses = [
        InteractionPose(0.75, 0.9, 0.0, 90.0, 0.0, True),
        InteractionPose(99.0, 0.9, 99.0, 0.0, 0.0, True),
    ]
    chosen = choose_best_interaction_pose(
        poses,
        reachable,
        {"x": 0.0, "y": 0.9, "z": 0.0},
        0.0,
        0.0,
        NavigationConfig(grid_size=0.25),
    )
    assert chosen is not None
    assert (chosen.x, chosen.z) == (0.75, 0.0)


def test_training_alias_and_levels_are_backward_compatible():
    assert _curriculum_from_config({}, "open_cross,pickup") == {"door": 1.0, "pickup": 1.0}
    args = build_parser().parse_args(["--episodes", "5", "--curriculum", "open_cross", "--level", "4"])
    assert args.level == 4


def test_navigation_level_zero_uses_short_pairs():
    import random
    reachable = [{"x": i * 0.25, "y": 0.9, "z": 0.0} for i in range(10)]
    pair = _navigation_pair_for_level(random.Random(3), reachable, 0, 0.25)
    assert pair is not None
    from agent_core.navigation import planar_distance
    assert 0.225 <= planar_distance(*pair) <= 0.525


def test_staged_skill_stats_are_tracked_per_curriculum_level(tmp_path: Path):
    lib = EmbodiedSkillLibrary(tmp_path / "skills.json", agent_label="A")
    lib.training_level = 2
    lib.record("door_approach", success=True, primitive_steps=5, object_type="Door")
    lib.record("door_approach", success=False, primitive_steps=7, object_type="Door")
    lib.training_level = 3
    lib.record("door_approach", success=True, primitive_steps=9, object_type="Door")

    assert lib.stats("door_approach", "Door").attempts == 3
    assert lib.level_stats("door_approach", 2).attempts == 2
    assert lib.level_stats("door_approach", 2).successes == 1
    assert lib.level_stats("door_approach", 3).attempts == 1
    assert lib.level_stats("door_approach", 3).successes == 1

    lib.save()
    reloaded = EmbodiedSkillLibrary(tmp_path / "skills.json", agent_label="A")
    assert reloaded.level_stats("door_approach", 2).attempts == 2
    assert reloaded.level_stats("door_approach", 3).successes == 1
