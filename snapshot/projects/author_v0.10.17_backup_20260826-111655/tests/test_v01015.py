from __future__ import annotations

import json
from types import SimpleNamespace

import agent_core
import agent_core.training as training


def test_release_version_is_v01015():
    assert agent_core.__version__ == "0.10.17"


def test_embodied_replay_schema_and_run_id(tmp_path):
    path = tmp_path / "20260826-120000" / "worker_00" / "embodied_replay.jsonl"
    path.parent.mkdir(parents=True)
    pose = SimpleNamespace(position={"x": 0.0, "y": 0.9, "z": 0.0}, rotation={"y": 0.0})
    world = SimpleNamespace(pose=pose, objects={})
    training._append_embodied_transition(
        path, scene="S", agent="A", episode=1, curriculum_skill="pickup",
        stage_skill="pickup_approach", curriculum_level=4,
        action=training.Action("NavigateTo", {}), before=world, after=world,
        success=True, verification={}, primitive_steps=1, primitive_actions=["MoveAhead"]
    )
    row = json.loads(path.read_text().strip())
    assert row["schema"] == "author.embodied.v01017"
    assert row["run_id"] == "20260826-120000"
    assert row["worker"] == "worker_00"


def test_pose_goal_replans_around_dynamic_blocker(monkeypatch):
    import agent_core.thor_adapter as thor
    from agent_core.interaction_pose import InteractionPose
    from agent_core.models import Action

    start = {"x": 0.0, "y": 0.9, "z": 0.0}
    blocked = {"x": 0.25, "y": 0.9, "z": 0.0}
    alternate = {"x": 0.0, "y": 0.9, "z": 0.25}
    goal = {"x": 0.25, "y": 0.9, "z": 0.25}
    pose = InteractionPose(x=goal["x"], y=0.9, z=goal["z"], rotation=0.0, horizon=0.0)

    def state(position, success=True, error=None):
        return SimpleNamespace(
            pose=SimpleNamespace(position=dict(position), rotation={"y": 0.0}, horizon=0.0),
            last_action_success=success,
            last_error=error,
        )

    adapter = SimpleNamespace(
        config=SimpleNamespace(grid_size=0.25, rotate_step_degrees=90.0),
        _state=state(start),
    )
    adapter.observe = lambda: adapter._state
    adapter.raw_event = lambda: object()

    def fake_path(rows, *_args, **_kwargs):
        keys = {(round(r["x"] / 0.25), round(r["z"] / 0.25)) for r in rows}
        if (1, 0) in keys:
            return [start, blocked, goal]
        return [start, alternate, goal]

    monkeypatch.setattr(thor, "shortest_grid_path", fake_path)
    monkeypatch.setattr(thor, "path_to_actions", lambda path, **kwargs: [Action("MoveAhead", {})])
    monkeypatch.setattr(thor, "_look_actions", lambda *_args, **_kwargs: [])

    calls = {"n": 0}
    def fake_exec(_adapter, primitives, _budget):
        calls["n"] += 1
        if calls["n"] == 1:
            adapter._state = state(start, False, "Discovery_Book is blocking Agent 1 from moving by (0,0,0.25).")
            return thor.ThorExecution(object(), adapter._state, 1, ["MoveAhead"], [])
        adapter._state = state(goal, True, None)
        return thor.ThorExecution(object(), adapter._state, 1 if primitives else 0, [a.kind for a in primitives], [])

    monkeypatch.setattr(thor, "_execute_primitives", fake_exec)
    result = thor._execute_pose_goal(adapter, pose, [start, blocked, alternate, goal], 20)
    assert result.state.last_action_success is True
    assert any(row.get("kind") == "dynamic_obstacle_replan" for row in (result.auxiliary_skills or []))
    assert calls["n"] >= 2
