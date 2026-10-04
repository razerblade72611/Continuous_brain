"""The live viewer (exploration lane): watching never changes the life; operator actions do what they say."""
import json
import threading
import urllib.request
from pathlib import Path

import pytest

from continuous_brain_individual.individual import Individual, IndividualConfig, first_info
from continuous_brain_individual.viewer import LiveSession, make_handler, Runner
from continuous_brain_organs.language import LanguageOrgan
from continuous_brain_world.env import W1Config, W1Env

REPO = Path(__file__).resolve().parents[1]
WEIGHTS = REPO / "runs/explore/vision/tile_decoder_v1.pt"
CARD = REPO / "continuous_brain_organs/cards/tile_decoder_v1.json"
SEED = 71003


def _vision():
    if WEIGHTS.exists():
        from continuous_brain_organs.vision import VisionOrgan
        return VisionOrgan(WEIGHTS, CARD)
    return None


VISION = _vision()
ARM = {"perception": "organ" if VISION is not None else "oracle"}


def _unwatched(n):
    ind = Individual(IndividualConfig(perception=ARM["perception"]), VISION, LanguageOrgan.from_config({"backend": "dev"}))
    env = W1Env(W1Config())
    obs = env.reset(SEED)
    info = first_info(env)
    actions = []
    for _ in range(n):
        a, u = ind.decide(obs, info)
        actions.append(a)
        obs, _, done, info = env.step(a, u)
        if done:
            break
    return actions, ind.state_hash()


def _session(tmp_path):
    s = LiveSession(VISION, {"backend": "dev"}, tmp_path / "lives")
    s.new_life(SEED, ARM)
    return s


def test_watching_does_not_change_the_life(tmp_path):
    actions, h = _unwatched(160)
    s = _session(tmp_path)
    for i in range(160):
        s.step()
        if i % 7 == 0:                       # everything the UI asks for, interleaved with the life
            s.frame()
            s.view_jpeg(256)
            s.meta()
    assert [r["a"] for r in s.records] == actions
    assert s.ind.state_hash() == h


def test_save_to_disk_and_resume_continue_exactly(tmp_path):
    s = _session(tmp_path)
    for _ in range(80):
        s.step()
    snap = s.save("t")
    for _ in range(60):
        s.step()
    straight, h = [r["a"] for r in s.records][80:], s.ind.state_hash()
    s2 = LiveSession(VISION, {"backend": "dev"}, tmp_path / "lives")
    s2.resume(snap)
    for _ in range(60):
        s2.step()
    assert [r["a"] for r in s2.records] == straight and s2.ind.state_hash() == h
    assert Path(str(snap) + ".world.json").exists() and Path(str(snap) + ".viewer.json").exists()
    assert s.save("t") != snap                          # a second save is a new snapshot, never an overwrite


def test_erase_bookmark_and_back(tmp_path):
    s = _session(tmp_path)
    for _ in range(120):
        s.step()
    known = s.frame()["memory"]["known"]
    s.mark()
    s.erase("episodic_memory")
    s.step()
    assert s.frame()["memory"]["known"] < known / 3          # it remembers only its view now
    assert "erased episodic_memory.state" in s.operator[-1]
    s.back()
    assert s.frame()["memory"]["known"] == known and s.ind.rt.tick == 119
    with pytest.raises(ValueError):
        s.erase("perception")


def test_the_http_api(tmp_path):
    from http.server import ThreadingHTTPServer
    s = _session(tmp_path)
    runner = Runner(s)
    runner.start()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(s, runner))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"

    def post(body):
        req = urllib.request.Request(base + "/api/control", json.dumps(body).encode(), {"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req).read())
    try:
        assert b"Continuous Brain" in urllib.request.urlopen(base + "/").read()
        meta = json.loads(urllib.request.urlopen(base + "/api/meta").read())
        assert meta["status"] == "ENGINEERING_ONLY" and len(meta["units"]) == 34
        assert post({"cmd": "step", "n": 30})["ok"]
        f = json.loads(urllib.request.urlopen(base + "/api/frame?since=10").read())
        assert f["tick"] == 29 and [r["t"] for r in f["records"]] == list(range(11, 30))
        assert len(f["records"][0]["u"]) == 34
        assert urllib.request.urlopen(base + "/api/view.jpg").read()[:2] == b"\xff\xd8"
        bad = urllib.request.Request(base + "/api/control", b'{"cmd": "nope"}')
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(bad)
    finally:
        runner.stop_flag = True
        httpd.shutdown()
