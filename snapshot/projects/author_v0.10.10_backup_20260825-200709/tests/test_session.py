from agent_core.session import SessionController, is_structured_verifiable_task


def test_session_starts_autonomous_then_accepts_plain_language_task():
    controller = SessionController()
    assert controller.state.mode == "autonomous"
    msg = controller.handle("Pick up the apple and put it in the sink.")
    assert controller.state.mode == "task"
    assert controller.state.active_task == "Pick up the apple and put it in the sink."
    assert "Task assigned" in msg


def test_roam_command_returns_to_autonomous_mode():
    controller = SessionController("Open the fridge.")
    controller.handle("/roam")
    assert controller.state.mode == "autonomous"
    assert controller.state.active_task is None


def test_quit_command_requests_exit():
    controller = SessionController()
    controller.handle("/quit")
    assert controller.state.quit_requested


def test_structured_tasks_use_hidden_simulator_verifier():
    assert is_structured_verifiable_task("Pick up the Apple and put it on the CounterTop.")
    assert is_structured_verifiable_task("Open the fridge.")
    assert not is_structured_verifiable_task("Look around and tell me what you think.")
