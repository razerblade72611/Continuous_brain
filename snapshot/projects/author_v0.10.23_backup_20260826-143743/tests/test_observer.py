from types import SimpleNamespace

import numpy as np

from agent_core.observer import ObserverDashboard


class FakeThor:
    def __init__(self, value: int):
        frame = np.full((24, 32, 3), value, dtype=np.uint8)
        self._event = SimpleNamespace(frame=frame)

    def raw_event(self):
        return self._event


class FakeLoop:
    def __init__(self, value: int, decision_index: int = 1, steps: int = 2):
        self.thor = FakeThor(value)
        self.decision_index = decision_index
        self.executor = SimpleNamespace(total_steps=steps)


def test_dashboard_snapshot_contains_two_independent_panels_without_private_state():
    dash = ObserverDashboard({0: "A", 1: "B"})
    loops = {0: FakeLoop(20), 1: FakeLoop(220)}
    dash.refresh_frames(loops)
    dash.update_agent(
        0,
        loops[0],
        mode="autonomous",
        task=None,
        record={
            "say": "I see a refrigerator.",
            "attention": "fridge",
            "pace": "slow",
            "semantic_actions": [{"kind": "Observe", "target": None, "params": {}}],
            "body_feedback": {"decision_success": True, "distance_moved": 0.0},
            "used_vision": True,
            "fallback": False,
            "spoken": [],
        },
    )
    dash.update_agent(
        1,
        loops[1],
        mode="task",
        task="Open the refrigerator.",
        record={
            "say": "I am facing a refrigerator.",
            "attention": "refrigerator",
            "pace": "normal",
            "semantic_actions": [{"kind": "Face", "target": "refrigerator", "params": {}}],
            "body_feedback": {"decision_success": True, "yaw_change_degrees": 90.0},
            "used_vision": True,
            "fallback": False,
            "spoken": ["Hello."],
        },
        heard=[{"speaker": "A", "text": "Hi.", "distance_m": 1.2}],
    )

    snap = dash.snapshot()
    assert [a["label"] for a in snap["agents"]] == ["A", "B"]
    assert snap["agents"][0]["report"] == "I see a refrigerator."
    assert snap["agents"][1]["task"] == "Open the refrigerator."
    assert snap["agents"][0]["image_data_uri"].startswith("data:image/jpeg;base64,")
    assert snap["agents"][1]["image_data_uri"].startswith("data:image/jpeg;base64,")

    serialized = repr(snap)
    for forbidden in ("chain_of_thought", "private_memory", "system_prompt", "object_catalog"):
        assert forbidden not in serialized


def test_dashboard_fallback_is_visible_in_event_log():
    dash = ObserverDashboard({0: "A", 1: "B"})
    loop = FakeLoop(30)
    dash.update_agent(
        0,
        loop,
        mode="autonomous",
        task=None,
        record={
            "say": "LLM command failed: JSONDecodeError",
            "semantic_actions": [{"kind": "Observe", "target": None, "params": {}}],
            "fallback": True,
        },
    )
    snap = dash.snapshot()
    assert snap["agents"][0]["fallback"] is True
    assert any(e["kind"] == "fallback" for e in snap["events"])


def test_dashboard_http_routes_are_read_only_and_render_both_agents():
    import json
    from urllib.error import HTTPError
    from urllib.request import Request, urlopen

    dash = ObserverDashboard({0: "A", 1: "B"}, port=0, refresh_ms=250)
    dash.start()
    try:
        with urlopen(dash.url + "/", timeout=2) as response:
            assert response.status == 200
            text = response.read().decode("utf-8")
        assert "AI2-THOR Social Observer" in text
        assert "not hidden chain-of-thought" in text
        assert "250" in text

        with urlopen(dash.url + "/api/state", timeout=2) as response:
            assert response.status == 200
            body = json.loads(response.read().decode("utf-8"))
        assert [a["label"] for a in body["agents"]] == ["A", "B"]

        try:
            urlopen(Request(dash.url + "/api/state", method="POST"), timeout=2)
        except HTTPError as exc:
            assert exc.code == 405
        else:
            raise AssertionError("POST must not be accepted by the read-only observer")
    finally:
        dash.stop()


def test_refresh_frames_updates_both_cameras_from_current_world():
    dash = ObserverDashboard({0: "A", 1: "B"})
    loops = {0: FakeLoop(10), 1: FakeLoop(240)}
    dash.refresh_frames(loops)
    first = dash.snapshot()
    a_img = first["agents"][0]["image_data_uri"]
    b_img = first["agents"][1]["image_data_uri"]
    assert a_img and b_img and a_img != b_img


def test_dashboard_exposes_peer_camera_cone_and_hearing_telemetry_without_private_state():
    dash = ObserverDashboard({0: "A", 1: "B"})
    loop = FakeLoop(80)
    dash.update_agent(
        0,
        loop,
        mode="autonomous",
        task=None,
        record={
            "say": "I see a metal rack.",
            "attention": "metal rack",
            "semantic_actions": [{"kind": "Observe", "target": None, "params": {}}],
            "body_feedback": {"recent_progress": {"total_distance_moved_m": 0.0, "stagnant": True}},
        },
        peer_presence=[{
            "peer_id": 1,
            "peer_label": "B",
            "distance_m": 1.6,
            "bearing_degrees": 8.0,
            "in_camera_cone": True,
            "within_hearing_range": True,
        }],
    )
    snap = dash.snapshot()
    peer = snap["agents"][0]["peer_presence"][0]
    assert peer["peer_label"] == "B"
    assert peer["in_camera_cone"] is True
    assert peer["within_hearing_range"] is True
    assert snap["agents"][0]["peer_reference_detected"] is False
    assert snap["agents"][0]["peer_sensor_disagreement"] is True
    assert "private_memory" not in repr(snap)
