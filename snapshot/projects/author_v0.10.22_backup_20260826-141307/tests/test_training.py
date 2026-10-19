from __future__ import annotations

from pathlib import Path

from agent_core.training import _split_episodes, build_parser, main


def test_training_episode_split():
    assert _split_episodes(10, 3) == [4, 3, 3]
    assert _split_episodes(2, 8) == [1, 1]


def test_training_parser_supports_reset_memory():
    args = build_parser().parse_args(["--reset-memory", "--episodes", "5", "--workers", "2"])
    assert args.reset_memory is True
    assert args.episodes == 5
    assert args.workers == 2


def test_training_dry_run_can_clear_state_without_unity(tmp_path: Path, capsys):
    project = tmp_path / "project"
    project.mkdir()
    config = project / "config.yaml"
    config.write_text(
        """
scene: FloorPlan1
grid_size: 0.25
rotate_step_degrees: 90
visibility_distance: 1.5
field_of_view: 90
simulation:
  local_executable_path: null
learning:
  state_dir: state
  memory_max_records: 100
  replay_max_records: 1000
training:
  episodes: 3
  workers: 1
""".strip()
        + "\n",
        encoding="utf-8",
    )
    state = project / "state"
    state.mkdir()
    (state / "old.txt").write_text("old", encoding="utf-8")

    rc = main(
        [
            "--config",
            str(config),
            "--state-dir",
            str(state),
            "--episodes",
            "3",
            "--reset-memory",
            "--dry-run",
        ]
    )
    assert rc == 0
    assert state.exists()
    assert not (state / "old.txt").exists()
    output = capsys.readouterr().out
    assert '"llm_calls": 0' in output
    assert '"episodes": 3' in output
