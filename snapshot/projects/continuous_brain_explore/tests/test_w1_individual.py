"""The W1 individual (exploration lane): store authority, persistence, the body-sense boundary, ablations.

Uses the trained vision organ when its weights are present (runs/explore/vision/tile_decoder_v1.pt, gitignored);
otherwise the perception=oracle arm, so the store/persistence properties are tested either way."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from continuous_brain_individual import codec, ledger, memory as mem
from continuous_brain_individual.body import BODY_KEYS, GuardedInfo, PrivilegedRead, body_sense
from continuous_brain_individual.individual import Individual, IndividualConfig, first_info
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
PERCEPTION = "organ" if VISION is not None else "oracle"


def _new(cfg=None, seed=SEED):
    cfg = cfg or IndividualConfig(perception=PERCEPTION)
    ind = Individual(cfg, VISION, LanguageOrgan.from_config({"backend": "dev"}))
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


# ---------------------------------------------------------------------------------------------------- codec
def test_codec_round_trips_exactly_and_canonically():
    obj = {"a": (1, 2, [3, (4,)]), "s": {3, 1, 2}, "d": {(1, 2): "x", 5: [None, True]},
           "arr": np.arange(12, dtype=np.int8).reshape(3, 4), "f": 0.25, "n": np.int64(7), "e": {}}
    back = codec.decode(codec.encode(obj))
    assert back["a"] == (1, 2, [3, (4,)]) and back["s"] == {1, 2, 3} and back["d"] == {(1, 2): "x", 5: [None, True]}
    assert np.array_equal(back["arr"], obj["arr"]) and back["arr"].dtype == np.int8 and back["n"] == 7
    assert codec.encode(back) == codec.encode(obj)                         # canonical: same object, same bytes
    assert codec.decode(b"") is None                                       # an erased channel
    with pytest.raises(ValueError):
        codec.encode({"__t": 1})


# ----------------------------------------------------------------------------------------------- body sense
def test_the_individual_can_read_only_its_body():
    env = W1Env(W1Config())
    env.reset(SEED)
    info = first_info(env)
    g = GuardedInfo(info)
    body = g.body()
    assert set(body) == set(BODY_KEYS)
    for key in ("privileged", "events", "material_map"):
        with pytest.raises(PrivilegedRead):
            g[key]
    assert body == body_sense(info)


def test_the_individual_never_touches_privileged_state_outside_its_body(monkeypatch):
    """Run a life where every privileged field other than the body sense is poisoned: same behaviour."""
    if VISION is None:
        pytest.skip("needs the trained vision organ (the oracle arm reads the current view by design)")
    ind, env, obs, info = _new()
    clean, _, _ = _live(ind, env, obs, info, 150)
    ind2, env2, obs2, info2 = _new()
    real_step = env2.step

    def poisoned_step(a, u=None):
        o, r, d, i = real_step(a, u)
        p = i["privileged"]
        keep = {k: p[k] for k in BODY_KEYS}
        i2 = {"privileged": {**{k: "POISON" for k in p}, **keep}, "events": "POISON"}
        return o, r, d, i2
    env2.step = poisoned_step
    p = info2["privileged"]
    info2 = {"privileged": {**{k: "POISON" for k in p}, **{k: p[k] for k in BODY_KEYS}}}
    poisoned, _, _ = _live(ind2, env2, obs2, info2, 150)
    assert poisoned == clean


# ---------------------------------------------------------------------------------------------- the store
def test_every_faculty_keeps_its_state_in_the_store():
    ind, env, obs, info = _new()
    _live(ind, env, obs, info, 60)
    channels = set(ind.rt.store.names())
    for slot in ("episodic_memory", "goal_genesis", "self_state", "metacognition", "skills", "autobiography"):
        assert f"{slot}.state" in channels
        assert ind.read_state(slot) is not None
    assert not ind.rt.scientific                                           # exploration: never scientific
    m = ind.read_state("episodic_memory")
    assert m["steps"] == 60 and (m["map"] >= 0).sum() > 0


def test_no_hidden_state_fresh_hosts_from_the_stored_state_behave_identically():
    """Rebuild the whole individual (new hosts, new organ objects) from its saved state every 25 steps: the life
    is identical to the uninterrupted one, so nothing that matters lives in Python between ticks."""
    ind, env, obs, info = _new()
    straight, _, _ = _live(ind, env, obs, info, 150)
    ind2, env2, obs2, info2 = _new()
    rebuilt = []
    for chunk in range(6):
        t, obs2, info2 = _live(ind2, env2, obs2, info2, 25)
        rebuilt += t
        saved = copy.deepcopy(ind2.save())
        ind2 = Individual(IndividualConfig(perception=PERCEPTION), VISION, LanguageOrgan.from_config({"backend": "dev"}))
        ind2.restore(saved)
    assert rebuilt == straight


def test_persistence_save_restore_continues_bit_exactly():
    ind, env, obs, info = _new()
    _, obs, info = _live(ind, env, obs, info, 80)
    saved, world = copy.deepcopy(ind.save()), env.snapshot()
    a, _, _ = _live(ind, env, obs, info, 70)
    h = ind.state_hash()
    ind2 = Individual(IndividualConfig(perception=PERCEPTION), VISION, LanguageOrgan.from_config({"backend": "dev"}))
    ind2.restore(saved)
    env2 = W1Env(W1Config())
    env2.restore(world)
    b, _, _ = _live(ind2, env2, obs, info, 70)
    assert a == b and ind2.state_hash() == h


def test_erasing_the_stored_memory_makes_the_individual_forget():
    """ERASE episodic_memory.state at tick 100 (a store intervention applied right after the memory host writes):
    the store then holds no memory at all; at the next tick the individual starts remembering from nothing (only
    the current view), and its life diverges from the untouched twin's."""
    ind, env, obs, info = _new()
    normal, _, _ = _live(ind, env, obs, info, 160)
    cfg = IndividualConfig(perception=PERCEPTION, interventions=[
        {"kind": "erase", "target": "episodic_memory.state", "start_tick": 100, "end_tick": 101}])
    ind2, env2, obs2, info2 = _new(cfg)
    t1, obs2, info2 = _live(ind2, env2, obs2, info2, 100)               # ticks 0..99
    known_before = int((ind2.read_state("episodic_memory")["map"] >= 0).sum())
    t2, obs2, info2 = _live(ind2, env2, obs2, info2, 1)                 # tick 100: erased right after the write
    assert ind2.read_state("episodic_memory") is None
    t3, obs2, info2 = _live(ind2, env2, obs2, info2, 1)                 # tick 101: remembering from nothing
    after = ind2.read_state("episodic_memory")
    assert t1 == normal[:100]
    assert after["steps"] == 1 and int((after["map"] >= 0).sum()) <= mem.ROWS * mem.COLS < known_before
    t4, _, _ = _live(ind2, env2, obs2, info2, 58)
    assert t1 + t2 + t3 + t4 != normal                                  # a different life from then on
    assert len(ind2.rt.interventions.records) == 1


def test_erasing_the_ledger_makes_it_forget_its_commitments():
    ind, env, obs, info = _new()
    _, obs, info = _live(ind, env, obs, info, 400)
    led = ind.read_state("goal_genesis")
    if not led["commitments"]:
        pytest.skip("no request heard in 400 steps on this seed")
    cfg = IndividualConfig(perception=PERCEPTION, interventions=[
        {"kind": "erase", "target": "goal_genesis.state", "start_tick": 400, "end_tick": 401}])
    ind2, env2, obs2, info2 = _new(cfg)
    _, obs2, info2 = _live(ind2, env2, obs2, info2, 401)
    assert ind2.read_state("goal_genesis") is None                       # erased in the store
    _, obs2, info2 = _live(ind2, env2, obs2, info2, 1)
    after = ind2.read_state("goal_genesis")
    assert not (set(after["commitments"]) & set(led["commitments"]))    # what it had promised is gone
    assert after["requests"] == [] or after["requests"][0]["issue_step"] > 400


# ------------------------------------------------------------------------------------------------ faculties
def test_memory_integrates_only_confident_perception_and_snapshots_each_day():
    tiles = {"material": np.full((7, 9), mem.MAT["grass"], np.int8), "object": np.zeros((7, 9), np.int8),
             "material_conf": np.full((7, 9), 0.99, np.float32), "object_conf": np.ones((7, 9), np.float32)}
    tiles["material"][0, 0] = mem.MAT["water"]
    tiles["material_conf"][0, 1] = 0.5                                    # unsure: not remembered as seen
    tiles["object"][3, 5] = mem.OBJ["cow"]
    body = {"player_pos": [20, 20], "achievements": {"eat_cow": 0}, "sleeping": False}
    st = mem.integrate(None, tiles, body, 10)
    assert st["map"][16, 17] == mem.MAT["water"] and st["seen"][17, 17] == -1
    assert st["last_seen"]["water"] == {"step": 10, "positions": [[16, 17]]}
    assert st["last_seen"]["cow"]["positions"] == [[21, 20]]
    assert st["day_snap"] == {"0": {"eat_cow": 0}}
    assert mem.counts_today(st, {"achievements": {"eat_cow": 2}}, 20) == {"eat_cow": 2}
    st2 = mem.integrate(st, {**tiles, "object": np.zeros((7, 9), np.int8)}, {**body, "player_pos": [40, 40]}, 11)
    assert any(o["kind"] == "cow" for o in mem.believed_objects(st2))       # out of view: still believed there
    st3 = mem.integrate(st2, {**tiles, "object": np.zeros((7, 9), np.int8)}, body, 12)
    assert not any(o["kind"] == "cow" for o in mem.believed_objects(st3))   # back in view and gone: forgotten
    none = mem.integrate(st, tiles, {**body, "player_pos": [40, 40]}, 11, mode="none")
    assert none["map"][16, 17] == -1                                       # the no-memory ablation forgets


def test_the_ledger_follows_what_was_heard():
    org = LanguageOrgan.from_config({"backend": "dev"})
    ctx = {"step": 100, "inventory": {"wood": 1}, "achievements": {}, "player_pos": [5, 5], "commitments": {}}
    evs = org.interpret("[o1] Please gather 3 wood within 200 steps.", ctx)
    st = ledger.apply_events(None, evs, 100)
    act = ledger.active(st, 101)
    assert [o["id"] for o in act] == ["o1"] and act[0]["spec"]["target"] == 4 and act[0]["deadline"] == 300
    st = ledger.apply_events(st, org.interpret("[o1] Never mind the wood.", {**ctx, "step": 150,
                                                                          "commitments": {"o1": evs[0]}}), 150)
    assert st["commitments"]["o1"]["status"] == "cancelled" and ledger.active(st, 151) == []
    assert st["cancellations"] == [{"id": "o1", "step": 150}]
    st = ledger.apply_events(ledger.apply_events(None, evs, 100), [], 306)
    assert st["commitments"]["o1"]["status"] == "expired"


@pytest.mark.parametrize("arm", [IndividualConfig(memory="none"), IndividualConfig(language="none"),
                                 IndividualConfig(perception="oracle")])
def test_every_ablation_arm_runs(arm):
    if arm.perception == "organ" and VISION is None:
        arm.perception = "oracle"
    ind, env, obs, info = _new(arm)
    _live(ind, env, obs, info, 40)
    if arm.language == "none":
        assert ind.read_state("goal_genesis")["commitments"] == {}
