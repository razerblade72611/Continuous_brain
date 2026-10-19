import json
from urllib.request import urlopen

from continuous_brain.visualizer import LiveVisualizer


def test_visualizer_state_and_controls():
    v = LiveVisualizer(port=0, trail_length=20)
    v.update({"step": 1, "agent": [0.1, -0.2], "target": [0.5, 0.5]})
    s = v.snapshot()
    assert s["step"] == 1
    assert s["trail"] == [[0.1, -0.2]]
    assert s["paused"] is False
    v.set_paused(True)
    v.set_speed(4.0)
    assert v.paused is True
    assert v.speed == 4.0


def test_visualizer_http_server():
    v = LiveVisualizer(port=0)
    v.update({"step": 9, "agent": [0.0, 0.0], "target": [0.2, 0.2]})
    url = v.start()
    try:
        with urlopen(url + "/health", timeout=2) as r:
            assert r.read() == b"ok\n"
        with urlopen(url + "/state", timeout=2) as r:
            state = json.loads(r.read())
        assert state["step"] == 9
        with urlopen(url, timeout=2) as r:
            html = r.read().decode("utf-8")
        assert "Continuous Brain v0.4.3" in html
        assert "world-model prediction" in html
        assert "waypoint" in html.lower()
    finally:
        v.close()
