"""Gate 7 (exploration lane): an organ swap keeps the same individual.  The swap carries every channel, the clock and
the life exactly (same state hash); only the organ's record in the binding changes; a snapshot will not restore under
another organ; a rollback swap returns it to living exactly as its never-swapped twin; anything but the organ
differing is refused."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from continuous_brain_cb2.organ_swap import (OrganSwapError, OrganSwapLog, identity_digest, organ_provenance,
                                             swap_organ)
from continuous_brain_individual.individual import Individual, IndividualConfig, first_info
from continuous_brain_organs import vision as V
from continuous_brain_organs.language import LanguageOrgan
from continuous_brain_world.env import W1Config, W1Env

REPO = Path(__file__).resolve().parents[1]
WEIGHTS = REPO / "runs/explore/vision/tile_decoder_v1.pt"
CARD = REPO / "continuous_brain_organs/cards/tile_decoder_v1.json"
pytestmark = pytest.mark.skipif(not WEIGHTS.exists(), reason="needs the v1 vision organ's weights")
SEED = 75116


def _other_organ(tmp_path):
    """A second, different vision organ: a small decoder with its own weights and card."""
    torch.manual_seed(7)
    dec = V.TileDecoder(width=16, blocks=1)
    sha = V.save_weights(dec, tmp_path / "dummy.pt")
    card = json.loads(CARD.read_text())
    card["name"], card["weights"] = "tile_decoder_dummy", {**card.get("weights", {}), "sha256": sha}
    card["architecture"] = {**card["architecture"], "config": dec.config}
    (tmp_path / "dummy.json").write_text(json.dumps(card))
    return V.VisionOrgan(tmp_path / "dummy.pt", tmp_path / "dummy.json")


def _ind(vision, skills="routine"):
    return Individual(IndividualConfig(lifelong=True, skills=skills), vision, LanguageOrgan.from_config({"backend": "dev"}))


def _live(ind, env, obs, info, n):
    acts = []
    for _ in range(n):
        a, u = ind.decide(obs, info)
        acts.append(a)
        obs, _, done, info = env.step(a, u)
        if done:
            break
    return acts, obs, info


def test_a_swap_keeps_the_individual_and_a_rollback_lives_as_its_twin(tmp_path):
    v1, v2 = V.VisionOrgan(WEIGHTS, CARD), _other_organ(tmp_path)
    src = _ind(v1)
    env = W1Env(W1Config())
    obs = env.reset(SEED)
    _, obs, info = _live(src, env, obs, first_info(env), 80)
    twin, env_twin = copy.deepcopy(src.save()), copy.deepcopy(env)
    h0, d0 = src.state_hash(), identity_digest(src)

    dst = _ind(v2)
    with pytest.raises(Exception):                         # an organ is part of the binding: no silent restore
        dst.restore(copy.deepcopy(src.save()))
    dst = _ind(v2)
    log = OrganSwapLog(tmp_path / "organ_swaps.jsonl")
    ev = swap_organ(src, dst, "vision", {"note": "test v1 -> dummy"})
    log.append(ev)
    assert ev["parent_state_hash"] == h0 == ev["state_hash"] == dst.state_hash()
    assert identity_digest(dst) == d0 and dst.rt.tick == src.rt.tick and dst.life == src.life
    assert organ_provenance(dst, "vision")["weights_sha256"] == v2.weights_sha256 != v1.weights_sha256
    assert ev["from"]["weights_sha256"] == v1.weights_sha256 and ev["to"]["weights_sha256"] == v2.weights_sha256

    back = _ind(v1)
    rb = swap_organ(dst, back, "vision", {"kind": "organ_rollback", "note": "test"}, prev_event_sha=ev["event_sha"])
    log.append(rb)
    assert log.verify() == 2 and rb["prev_event_sha"] == ev["event_sha"] and back.state_hash() == h0
    ref = _ind(v1)
    ref.restore(twin)
    a1, _, _ = _live(back, env, obs, info, 60)
    a2, _, _ = _live(ref, env_twin, obs, info, 60)
    assert a1 == a2 and back.state_hash() == ref.state_hash()


def test_anything_but_the_organ_differing_is_refused(tmp_path):
    v1 = V.VisionOrgan(WEIGHTS, CARD)
    src = _ind(v1)
    env = W1Env(W1Config())
    obs = env.reset(SEED)
    _live(src, env, obs, first_info(env), 5)
    with pytest.raises(OrganSwapError):
        swap_organ(src, _ind(v1, skills="cb2"), "vision", {})          # a different skill binding is not an organ
    with pytest.raises(OrganSwapError):
        swap_organ(src, _ind(v1), "language", {})                      # no protocol for the language organ yet
