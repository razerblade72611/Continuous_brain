"""CB2-S1 consolidation and the learned skill in the store (exploration lane): a transplant keeps everything the
individual was except its skill faculty; the learned skill has no hidden state; events chain; rollback is exact."""
import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from continuous_brain_cb2.consolidation import ConsolidationLog, install
from continuous_brain_cb2.policy import PolicyConfig, SkillPolicy
from continuous_brain_cb2.skill_host import pack_weights
from continuous_brain_cb2.world_model import CoreConfig, PredictiveCore
from continuous_brain_individual.individual import Individual, IndividualConfig, first_info
from continuous_brain_organs.language import LanguageOrgan
from continuous_brain_world.env import W1Config, W1Env

REPO = Path(__file__).resolve().parents[1]
WEIGHTS = REPO / "runs/explore/vision/tile_decoder_v1.pt"
CARD = REPO / "continuous_brain_organs/cards/tile_decoder_v1.json"
TINY = CoreConfig(deter=64, hidden=64, stoch=8, discrete=8, cnn_depth=8, units=64)
SEED = 75111


def _vision():
    if WEIGHTS.exists():
        from continuous_brain_organs.vision import VisionOrgan
        return VisionOrgan(WEIGHTS, CARD)
    return None


VISION = _vision()
PERCEPTION = "organ" if VISION is not None else "oracle"


def _ind(skills="routine"):
    return Individual(IndividualConfig(perception=PERCEPTION, lifelong=True, skills=skills), VISION,
                      LanguageOrgan.from_config({"backend": "dev"}))


def _weights(arm="P-A", seed=0):
    torch.manual_seed(seed)
    core = PredictiveCore(TINY)
    return pack_weights(core, SkillPolicy(PolicyConfig(arm=arm, units=64, layers=1), core.feat_size))


def _live(ind, env, obs, info, n):
    acts = []
    for _ in range(n):
        a, u = ind.decide(obs, info)
        acts.append(a)
        obs, _, done, info = env.step(a, u)
        if done:
            break
    return acts, obs, info


def test_a_transplant_keeps_the_individual_and_changes_only_its_skill(tmp_path):
    src = _ind()
    env = W1Env(W1Config())
    obs = env.reset(SEED)
    _, obs, info = _live(src, env, obs, first_info(env), 120)
    before = {n: np.array(src.rt.store.read(n)) for n in src.rt.store.names() if not n.startswith("skills.")}
    dst = _ind("cb2")
    ev = install(src, dst, _weights(), {"note": "test"})
    assert ev["kind"] == "transplant" and ev["parent_state_hash"] == src.state_hash()
    assert dst.rt.tick == src.rt.tick and dst.life == src.life
    for n, v in before.items():
        assert np.array_equal(np.array(dst.rt.store.read(n)), v), n
    assert dst.read_state("skills")["lineage"][0]["event_sha"] == ev["event_sha"]
    # it lives on with the learned skill, and the same moment replays exactly from its saved state
    saved = copy.deepcopy(dst.save())
    env2 = copy.deepcopy(env)
    a1, _, _ = _live(dst, env, obs, info, 40)
    h1 = dst.state_hash()
    again = _ind("cb2")
    again.restore(saved)
    a2, _, _ = _live(again, env2, obs, info, 40)
    assert a1 == a2 and again.state_hash() == h1
    assert len(np.asarray(dst.rt.store.read("skills.core_state"))) == 4 * (64 + 8 * 8)


def test_installs_chain_and_rollback_is_exact(tmp_path):
    src = _ind()
    env = W1Env(W1Config())
    obs = env.reset(SEED)
    _, obs, info = _live(src, env, obs, first_info(env), 50)
    a = _ind("cb2")
    log = ConsolidationLog(tmp_path / "consolidations.jsonl")
    e1 = install(src, a, _weights(seed=1), {"round": 1})
    log.append(e1)
    snap = copy.deepcopy(a.save())
    _live(a, env, obs, info, 10)
    b = _ind("cb2")
    e2 = install(a, b, _weights(seed=2), {"round": 2})
    log.append(e2)
    assert e2["kind"] == "install" and e2["prev_event_sha"] == e1["event_sha"]
    assert [x["event_sha"] for x in b.read_state("skills")["lineage"]] == [e1["event_sha"], e2["event_sha"]]
    assert log.verify() == 2
    back = _ind("cb2")
    back.restore(snap)                                         # rollback: the snapshot the first event produced
    assert back.state_hash() == e1["state_hash"]
    rows = (tmp_path / "consolidations.jsonl").read_text().splitlines()
    (tmp_path / "consolidations.jsonl").write_text(rows[1] + "\n" + rows[0] + "\n")
    with pytest.raises(ValueError):
        log.verify()


def test_both_arms_act():
    for arm in ("P-A", "P-B"):
        src = _ind()
        env = W1Env(W1Config())
        obs = env.reset(SEED)
        _, obs, info = _live(src, env, obs, first_info(env), 5)
        dst = _ind("cb2")
        install(src, dst, _weights(arm), {})
        acts, _, _ = _live(dst, env, obs, info, 5)
        assert len(acts) == 5 and all(0 <= x < 17 for x in acts)
