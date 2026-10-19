"""The learned faculties and the lifelong individual (exploration lane): exact learning machinery, everything
learned lives in the store, a life ends and the same individual continues, frozen means frozen."""
import copy
from pathlib import Path

import numpy as np
import pytest

from continuous_brain_individual import learned as L
from continuous_brain_individual.features import DIM, NAMES
from continuous_brain_individual.individual import Individual, IndividualConfig, first_info
from continuous_brain_individual.learning import Adam, Layout, bce_grad, huber_grad, mlp_backward, mlp_forward, mlp_init
from continuous_brain_organs.language import LanguageOrgan
from continuous_brain_world.env import W1Config, W1Env

REPO = Path(__file__).resolve().parents[1]
WEIGHTS = REPO / "runs/explore/vision/tile_decoder_v1.pt"
CARD = REPO / "continuous_brain_organs/cards/tile_decoder_v1.json"
SEED = 73005


def _vision():
    if WEIGHTS.exists():
        from continuous_brain_organs.vision import VisionOrgan
        return VisionOrgan(WEIGHTS, CARD)
    return None


VISION = _vision()
PERCEPTION = "organ" if VISION is not None else "oracle"


def _cfg(**kw):
    return IndividualConfig(perception=PERCEPTION, skills="learned", lifelong=True, **kw)


def _new(cfg=None, seed=SEED):
    ind = Individual(cfg or _cfg(), VISION, LanguageOrgan.from_config({"backend": "dev"}))
    env = W1Env(W1Config())
    obs = env.reset(seed)
    return ind, env, obs, first_info(env)


def _live(ind, env, obs, info, n):
    trace = []
    for _ in range(n):
        a, u = ind.decide(obs, info)
        trace.append((a, u))
        obs, _, done, info = env.step(a, u)
        if done:
            break
    return trace, obs, info


# ------------------------------------------------------------------------------------------- the machinery
def test_backpropagation_matches_finite_differences():
    rng = np.random.default_rng(0)
    sizes = [7, 6, 5, 3]
    p = {k: v.astype(np.float64) for k, v in mlp_init(rng, sizes, out_scale=1.0).items()}
    for k in p:                     # nonzero biases: keep pre-activations off ReLU's kink at exactly 0
        if k.startswith("b"):
            p[k] = rng.standard_normal(p[k].shape) * 0.3
    x = rng.standard_normal((8, 7))
    w = rng.standard_normal((8, 3))
    out, cache = mlp_forward(p, x, 3)
    g = mlp_backward(p, cache, w.astype(np.float32), 3)
    for k in p:
        num = np.zeros_like(p[k])
        it = np.nditer(p[k], flags=["multi_index"])
        for _ in it:
            i = it.multi_index
            old = p[k][i]
            p[k][i] = old + 1e-4
            up = float((mlp_forward(p, x, 3)[0] * w).sum())
            p[k][i] = old - 1e-4
            dn = float((mlp_forward(p, x, 3)[0] * w).sum())
            p[k][i] = old
            num[i] = (up - dn) / 2e-4
        assert np.allclose(g[k], num, atol=2e-3), k


def test_losses_adam_and_layout():
    l, g = huber_grad(np.array([0.5, -3.0]))
    assert l == pytest.approx((0.125 + 2.5) / 2) and np.allclose(g, [0.25, -0.5])
    l, g = bce_grad(np.array([[0.0]]), np.array([[1.0]]))
    assert l == pytest.approx(np.log(2), rel=1e-5) and g[0, 0] == pytest.approx(-0.5)
    lay = Layout([("a", (2, 3)), ("b", (4,))])
    flat = lay.zeros()
    lay.view(flat, "a")[...] = 1
    assert lay.size == 10 and flat[:6].sum() == 6 and lay.view(flat, "b").sum() == 0
    p, m, v = {"w": np.ones(3, np.float32)}, {"w": np.zeros(3, np.float32)}, {"w": np.zeros(3, np.float32)}
    Adam(lr=0.1).step(p, {"w": np.ones(3, np.float32)}, m, v, 1)
    assert np.allclose(p["w"], 0.9, atol=1e-6)                   # Adam's first step moves by lr
    assert len(NAMES) == DIM


# ------------------------------------------------------------------------------------------- in the store
def test_learning_happens_and_lives_in_the_store(monkeypatch):
    monkeypatch.setattr(L, "Q_MIN", 8)                  # start learning sooner than in a real life, to test it
    monkeypatch.setattr(L, "WM_MIN", 16)
    ind, env, obs, info = _new()
    w0 = None
    _live(ind, env, obs, info, 1)
    w0 = (np.array(ind.rt.store.read("skills.weights")), np.array(ind.rt.store.read("world_model.weights")))
    _, obs, info = _live(ind, env, obs, info, 300)
    assert np.array_equal(w0[0], np.array(ind.rt.store.read("skills.weights")))   # choice waits for the life's end
    ind.die(obs, info)                                   # the life's returns are known: it consolidates
    sk, wm = ind.read_state("skills")["ctl"], ind.read_state("world_model")
    assert sk["decisions"] > 10 and wm["samples"] > 30 and sk["updates"] >= L.Q_CONSOLIDATE
    assert not np.array_equal(w0[0], np.array(ind.rt.store.read("skills.weights")))       # it changed itself
    assert not np.array_equal(w0[1], np.array(ind.rt.store.read("world_model.weights")))
    for ch in ("skills.weights", "skills.memory", "world_model.weights", "world_model.memory",
               "world_model.prediction", "skills.values"):
        assert ch in ind.rt.store.names()
    assert not ind.rt.scientific
    assert sk["updates"] > 0 and wm["updates"] > 0                  # it has started learning from its own life


def test_no_hidden_state_when_learning():
    ind, env, obs, info = _new()
    straight, _, _ = _live(ind, env, obs, info, 150)
    h = ind.state_hash()
    ind2, env2, obs2, info2 = _new()
    rebuilt = []
    for _ in range(6):
        t, obs2, info2 = _live(ind2, env2, obs2, info2, 25)
        rebuilt += t
        saved = copy.deepcopy(ind2.save())
        ind2 = Individual(_cfg(), VISION, LanguageOrgan.from_config({"backend": "dev"}))
        ind2.restore(saved)
    assert rebuilt == straight and ind2.state_hash() == h


def test_a_life_ends_and_the_same_individual_continues():
    ind, env, obs, info = _new()
    _, obs, info = _live(ind, env, obs, info, 200)
    ind.die(obs, info)                                         # its last moment (the driver says it is dying)
    learned_w = np.array(ind.rt.store.read("skills.weights"))
    fore_w = np.array(ind.rt.store.read("world_model.weights"))
    self_before = ind.read_state("self_state")
    auto = ind.read_state("autobiography")
    assert len(auto["lives"]) == 1 and "my life ended" in auto["days"][-1]["text"]
    assert ind.read_state("skills")["ctl"]["lives"] == 1
    ind.next_life()
    env2 = W1Env(W1Config())
    obs2 = env2.reset(SEED + 1)
    _live(ind, env2, obs2, first_info(env2), 3)
    mem = ind.read_state("episodic_memory")
    assert mem["steps"] == 3 and (mem["map"] >= 0).mean() < 0.03     # it remembers nothing of the old world
    assert ind.read_state("goal_genesis")["commitments"] == {}
    assert ind.read_state("self_state") == self_before                 # what it believes about itself persists
    assert ind.read_state("autobiography")["days"][0]["text"] == auto["days"][0]["text"]   # and its diary
    assert "woke up in a new world" in ind.read_state("autobiography")["today"][0]["text"]
    # what it learned crosses into the new life untouched (too few memories yet for an update in 3 ticks)
    assert ind.read_state("skills")["ctl"]["count"] < L.Q_MIN and ind.read_state("world_model")["count"] < L.WM_MIN
    assert np.array_equal(np.array(ind.rt.store.read("skills.weights")), learned_w)
    assert np.array_equal(np.array(ind.rt.store.read("world_model.weights")), fore_w)


def test_frozen_does_not_learn():
    ind, env, obs, info = _new()
    _, obs, info = _live(ind, env, obs, info, 300)
    saved = copy.deepcopy(ind.rt.store.state_dict())
    frozen = Individual(_cfg(learn=False), VISION, LanguageOrgan.from_config({"backend": "dev"}))
    frozen.rt.store.load_state_dict(copy.deepcopy(saved))
    frozen.rt.tick = int(saved["tick"])
    w = np.array(frozen.rt.store.read("skills.weights"))
    f = np.array(frozen.rt.store.read("world_model.weights"))
    _live(frozen, env, obs, info, 200)
    assert np.array_equal(np.array(frozen.rt.store.read("skills.weights")), w)
    assert np.array_equal(np.array(frozen.rt.store.read("world_model.weights")), f)


def test_erasing_skills_state_forgets_what_it_learned(monkeypatch):
    monkeypatch.setattr(L, "Q_MIN", 8)
    ind, env, obs, info = _new()
    _, obs, info = _live(ind, env, obs, info, 300)
    ind.die(obs, info)
    ind.next_life()
    env = W1Env(W1Config())
    obs = env.reset(SEED + 1)
    info = first_info(env)
    trained = np.array(ind.rt.store.read("skills.weights"))
    st = ind.rt.state_dict()
    st["store"]["current"]["skills.state"] = np.zeros(0, np.uint8)
    ind.rt.load_state_dict(st)
    ind.decide(obs, info)
    ctl = ind.read_state("skills")["ctl"]
    assert ctl["decisions"] <= 1 and ctl["updates"] == 0
    fresh = Individual(_cfg(), VISION, LanguageOrgan.from_config({"backend": "dev"}))
    fresh.life = ind.life
    env2 = W1Env(W1Config())
    fresh.decide(env2.reset(SEED + 1), first_info(env2))
    assert np.array_equal(np.array(ind.rt.store.read("skills.weights")), np.array(fresh.rt.store.read("skills.weights")))
    assert not np.array_equal(trained, np.array(fresh.rt.store.read("skills.weights")))


def test_always_choosing_its_routine_is_the_routine_individual(monkeypatch):
    monkeypatch.setattr(L, "KAPPA", 1e9)                     # never confident enough to leave its routine
    monkeypatch.setattr(L, "EPS0", 0.0)
    monkeypatch.setattr(L, "EPS_MIN", 0.0)
    a, _, _ = _live(*_new(IndividualConfig(perception=PERCEPTION)), 300)
    b, _, _ = _live(*_new(), 300)
    assert a == b
