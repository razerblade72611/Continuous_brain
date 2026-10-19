from agent_core.navigation import NavigationConfig, path_to_actions, shortest_grid_path


def test_shortest_grid_path_reaches_point_near_target():
    reachable = [
        {"x": 0.0, "y": 0.9, "z": 0.0},
        {"x": 0.25, "y": 0.9, "z": 0.0},
        {"x": 0.5, "y": 0.9, "z": 0.0},
        {"x": 0.75, "y": 0.9, "z": 0.0},
        {"x": 1.0, "y": 0.9, "z": 0.0},
    ]
    cfg = NavigationConfig(grid_size=0.25, interaction_distance=0.3)
    path = shortest_grid_path(
        reachable,
        {"x": 0.0, "y": 0.9, "z": 0.0},
        {"x": 1.2, "y": 1.0, "z": 0.0},
        cfg,
    )
    assert path[0]["x"] == 0.0
    assert path[-1]["x"] == 1.0


def test_path_to_actions_turns_and_moves():
    path = [
        {"x": 0.0, "y": 0.9, "z": 0.0},
        {"x": 0.25, "y": 0.9, "z": 0.0},
        {"x": 0.5, "y": 0.9, "z": 0.0},
    ]
    actions = path_to_actions(
        path,
        initial_yaw=0.0,
        target_position={"x": 0.75, "y": 1.0, "z": 0.0},
        config=NavigationConfig(),
    )
    kinds = [action.kind for action in actions]
    assert kinds[:3] == ["RotateRight", "MoveAhead", "MoveAhead"]
