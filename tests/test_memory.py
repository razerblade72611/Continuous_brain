import numpy as np
import pytest

from continuous_brain.memory import EpisodicStore


def _add(store, key, query="q", **kw):
    return store.add(query=query, prompt=query, response="r", t_sim=1.0, cycle_s=0.02,
                     features=np.zeros((4, 3)), key=np.asarray(key, np.float32),
                     wb_summary=np.zeros(8), wb_current=np.zeros(2), core_inputs=np.zeros((5, 2)), **kw)


def test_persists_across_reopen(tmp_path):
    s = EpisodicStore(tmp_path)
    eid = _add(s, [1, 0, 0])
    s.annotate(eid, outcome=0.7, target="the answer")
    s.mark_consolidated([eid], "sleep-1")
    reopened = EpisodicStore(tmp_path)
    ep = reopened.get(eid)
    assert ep.outcome == 0.7 and ep.target == "the answer" and ep.consolidated_in == ["sleep-1"]
    assert ep.arrays["core_inputs"].shape == (5, 2)


def test_log_is_append_only(tmp_path):
    s = EpisodicStore(tmp_path)
    eid = _add(s, [1, 0])
    first = s.log_path.read_text()
    s.annotate(eid, outcome=1.0)
    assert s.log_path.read_text().startswith(first)


def test_novelty_and_salience(tmp_path):
    s = EpisodicStore(tmp_path)
    a = _add(s, [1, 0, 0])
    b = _add(s, [1, 0, 0])            # duplicate: no novelty
    c = _add(s, [0, 1, 0])            # orthogonal: fully novel
    eps = {e.id: e for e in s.all()}
    assert eps[a].novelty == 1.0 and eps[b].novelty == pytest.approx(0.0, abs=1e-6)
    assert eps[c].novelty == pytest.approx(1.0, abs=1e-6)
    s.annotate(b, outcome=-2.0, explicit=True)
    assert s.all()[1].salience > s.all()[2].salience


def test_search_orders_by_similarity(tmp_path):
    s = EpisodicStore(tmp_path)
    near = _add(s, [1, 0.1, 0], query="near")
    _add(s, [0, 0, 1], query="far")
    hits = s.search(np.array([1, 0, 0], np.float32), k=2)
    assert hits[0][0].id == near and hits[0][1] > hits[1][1]
    assert s.search(np.array([1, 0, 0], np.float32), k=2, min_similarity=0.9)[0][0].id == near


def test_select_for_sleep_skips_consolidated(tmp_path):
    s = EpisodicStore(tmp_path)
    a, b = _add(s, [1, 0]), _add(s, [0, 1])
    s.mark_consolidated([a], "sleep-1")
    assert [e.id for e in s.select_for_sleep(10)] == [b]
