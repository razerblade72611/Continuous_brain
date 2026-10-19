"""r3-T diagnostic (exploration lane): the intent-watching library is the routine library, unchanged; an intent
policy starts as the policy it was copied from; a candidate whose policy reads intent acts the same way twice (install events carry a time, so whole-state hashes differ by design)."""
import copy
from pathlib import Path

import numpy as np
import torch

from continuous_brain_cb2.consolidation import install
from continuous_brain_cb2.intent import BRANCHES, INTENT_DIM, IntentShadow
from continuous_brain_cb2.policy import PolicyConfig, SkillPolicy
from continuous_brain_cb2.skill_host import pack_weights
from continuous_brain_cb2.world_model import CoreConfig, PredictiveCore
from continuous_brain_individual.individual import Individual, IndividualConfig, first_info
from continuous_brain_individual.skills import ACTIONS
from continuous_brain_organs.language import LanguageOrgan
from continuous_brain_world.env import W1Config, W1Env

REPO = Path(__file__).resolve().parents[1]
WEIGHTS = REPO / "runs/explore/vision/tile_decoder_v1.pt"
CARD = REPO / "continuous_brain_organs/cards/tile_decoder_v1.json"
TINY = CoreConfig(deter=64, hidden=64, stoch=8, discrete=8, cnn_depth=8, units=64)


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


def test_the_intent_shadow_is_the_routine():
    """On a routine's own life, the watched library chooses the routine's action at every step."""
    branches = set()
    for seed in (75111, 75112):
        ind, env = _ind(), W1Env(W1Config())
        obs, info = env.reset(seed), None
        info = first_info(env)
        shadow = IntentShadow(0)
        for _ in range(400):
            a, u = ind.decide(obs, info)
            name, _, vec = shadow(*IntentShadow.read(ind))
            assert ACTIONS[name] == a
            assert vec.shape == (INTENT_DIM,) and vec[:len(BRANCHES)].sum() == 1
            branches.add(BRANCHES[int(vec[:len(BRANCHES)].argmax())])
            obs, _, done, info = env.step(a, u)
            if done:
                break
    assert len(branches) >= 3


def test_an_intent_policy_starts_as_its_base():
    torch.manual_seed(0)
    core = PredictiveCore(TINY)
    base = SkillPolicy(PolicyConfig(arm="P-A", units=64, layers=2), core.feat_size)
    feat = torch.randn(2, 3, core.feat_size)
    body = torch.randn(2, 3, 23)
    feats = torch.randn(2, 3, 62)
    crop = torch.randint(-1, 19, (2, 3, 33 * 33))
    intent = torch.rand(2, 3, INTENT_DIM)
    want = base(feat, body, feats, crop)[0]
    for m in ("none", "branch", "full"):
        p = SkillPolicy.with_intent(base, m)
        assert torch.allclose(p(feat, body, feats, crop, intent)[0], want, atol=1e-6)
        q = SkillPolicy.from_state(copy.deepcopy(p.state_dict_full()))
        assert q.cfg.intent_mask == m
        assert torch.allclose(q(feat, body, feats, crop, intent)[0], want, atol=1e-6)


def test_a_candidate_reading_intent_acts_and_replays():
    torch.manual_seed(1)
    core = PredictiveCore(TINY)
    base = SkillPolicy(PolicyConfig(arm="P-A", units=64, layers=1), core.feat_size)
    pol = SkillPolicy.with_intent(base, "full")
    with torch.no_grad():                                       # make the intent columns matter
        pol.trunk[0].weight[:, -INTENT_DIM:] = torch.randn(pol.trunk[0].weight.shape[0], INTENT_DIM)
    w = pack_weights(core, pol)
    runs = []
    for _ in range(2):
        src = _ind()
        dst = _ind("cb2")
        install(src, dst, w, {})
        env = W1Env(W1Config())
        obs, info = env.reset(75113), None
        info = first_info(env)
        acts = []
        for _ in range(30):
            a, u = dst.decide(obs, info)
            acts.append(a)
            obs, _, done, info = env.step(a, u)
            if done:
                break
        from continuous_brain_individual import codec
        st = codec.unblob(dst.rt.store.read("skills.state"))
        assert st["shadow"] is not None                         # the routine's plan state lives in the store
        runs.append(acts)
    assert runs[0] == runs[1] and all(0 <= x < 17 for x in runs[0])


def test_the_organ_takes_commands_not_goals():
    import pytest
    from continuous_brain_cb2.organ import COMMANDS, ExecutorOrgan
    assert "routine" not in COMMANDS
    ind, env = _ind(), W1Env(W1Config())
    obs = env.reset(75114)
    info = first_info(env)
    a, u = ind.decide(obs, info)
    organ = ExecutorOrgan(0)
    with pytest.raises(ValueError):
        organ("routine", *IntentShadow.read(ind))
    name, tel = organ("explore", *IntentShadow.read(ind))
    assert name in ACTIONS and tel["command"] == "explore" and tel["status"].split("+")[0] in ("ok", "cannot")


def test_a_selector_candidate_delegates_every_action():
    from continuous_brain_cb2.organ import COMMANDS
    from continuous_brain_individual import codec
    torch.manual_seed(2)
    core = PredictiveCore(TINY)
    base = SkillPolicy(PolicyConfig(arm="P-A", units=64, layers=1), core.feat_size)
    sel = SkillPolicy.selector_from(base)
    assert sel.n_out == len(COMMANDS) and torch.equal(sel.trunk[0].weight, base.trunk[0].weight)
    src, dst = _ind(), _ind("cb2")
    install(src, dst, pack_weights(core, sel), {})
    env = W1Env(W1Config())
    obs = env.reset(75115)
    info = first_info(env)
    for i in range(25):
        a, u = dst.decide(obs, info)
        st = codec.unblob(dst.rt.store.read("skills.state"))
        assert st["delegation"]["provenance"] == "scripted-executor-organ"
        assert ACTIONS[st["delegation"]["action"]] == a and st["delegation"]["command"] in COMMANDS
        obs, _, done, info = env.step(a, u)
        if done:
            break
    assert sum(st["delegated"].values()) == i + 1
