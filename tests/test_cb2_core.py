"""CB2-S1 predictive core and its training data (exploration lane): step and scan are one dynamics; risk targets
look only forward and only inside a life; the core learns; a saved core is the same core."""
import numpy as np
import torch

from continuous_brain_cb2.data import BODY_DIM, N_RISK, RISK_H, risk_targets
from continuous_brain_cb2.world_model import CoreConfig, PredictiveCore

TINY = CoreConfig(deter=64, hidden=64, stoch=8, discrete=8, cnn_depth=8, units=64, lr=1e-3)


def _batch(B=2, T=12, seed=0):
    rng = np.random.default_rng(seed)
    first = np.zeros((B, T), np.float32)
    first[:, 0] = 1
    first[1, 7] = 1                                                    # a new life inside the window
    color = rng.integers(0, 256, (B, T, 1, 1, 3), dtype=np.uint8)                # learnable frames
    return {"image": np.broadcast_to(color, (B, T, 64, 64, 3)).copy(),
            "body": rng.random((B, T, BODY_DIM)).astype(np.float32),
            "prev_action": rng.integers(0, 17, (B, T)), "is_first": first,
            "reward": rng.random((B, T)).astype(np.float32), "cont": np.ones((B, T), np.float32),
            "risk": (rng.random((B, T, N_RISK)) < 0.2).astype(np.float32)}


def test_step_and_scan_are_one_dynamics():
    torch.manual_seed(0)
    core = PredictiveCore(TINY)
    b = _batch()
    post, _ = core.scan(b["image"], b["body"], b["prev_action"], b["is_first"])
    st = None
    for t in range(b["image"].shape[1]):
        st = core.step(st, b["prev_action"][:, t], b["image"][:, t], b["body"][:, t], b["is_first"][:, t])
        for k in ("deter", "stoch"):
            assert torch.allclose(st[k], post[k][:, t], atol=1e-5), (k, t)
    # a new life resets: the state at the reset does not depend on what came before it
    b2 = {k: v.copy() for k, v in b.items()}
    b2["image"][1, :7] = 0
    post2, _ = core.scan(b2["image"], b2["body"], b2["prev_action"], b2["is_first"])
    assert torch.allclose(post2["deter"][1, 7:], post["deter"][1, 7:], atol=1e-5)
    assert not torch.allclose(post2["deter"][1, 3], post["deter"][1, 3])


def test_the_core_learns_and_a_saved_core_is_the_same_core():
    torch.manual_seed(0)
    core = PredictiveCore(TINY)
    b = _batch(seed=1)
    first = core.train_step(b)["loss"]
    for _ in range(30):
        last = core.train_step(b)["loss"]
    assert last < first * 0.8
    again = PredictiveCore.from_state(core.state_dict_full())
    p1, _ = core.scan(b["image"], b["body"], b["prev_action"], b["is_first"])
    p2, _ = again.scan(b["image"], b["body"], b["prev_action"], b["is_first"])
    assert torch.equal(p1["deter"], p2["deter"])
    feats, acts, _ = core.imagine({k: v[:, -1] for k, v in p1.items()}, lambda f: torch.zeros(len(f), dtype=torch.long), 5)
    assert feats.shape == (2, 5, core.feat_size) and acts.shape == (2, 5)
    assert core.evaluate(b)["openloop_mse"] >= 0


def test_risk_targets_look_forward_inside_the_life():
    # life 0: 10 steps, dies; drink runs out at step 6.  life 1: 5 steps, ends by the step limit (not a death)
    life = np.r_[np.zeros(10), np.ones(5)].astype(np.int32)
    step = np.r_[np.arange(10), np.arange(9995, 10000)].astype(np.int32)
    done = np.zeros(15, bool)
    done[9] = done[14] = True
    inv = np.full((15, 16), 9, np.int16)
    inv[6:10, 2] = 0
    inv[8:10, 0] = 5                                                   # health 9 -> 5 at step 8
    r = risk_targets(life, step, done, inv)
    j25 = RISK_H.index(25)
    die, water, hurt = r[:, 0 * 3 + j25], r[:, 1 * 3 + j25], r[:, 3 * 3 + j25]
    assert die[:10].all() and not die[10:].any()                       # life 1 ended without dying
    assert water[:6].all() and water[6:9].all() and not water[9]       # next zero strictly after i
    assert hurt[:8].all() and not hurt[8:].any()
    assert r[9, 0 * 3 + j25] == 1 and r[10:].sum() == 0


def test_dreams_are_archived_as_dreamed_and_never_as_lived(tmp_path):
    from continuous_brain_cb2.archive import Archive, StreamWriter
    from continuous_brain_cb2.dream import dream
    from continuous_brain_cb2.policy import PolicyConfig, SkillPolicy

    class Fake:                                                         # a sampler over one synthetic batch
        def batch(self, B):
            return {k: v[:B] for k, v in _batch(B=max(B, 2)).items()}
    torch.manual_seed(0)
    core = PredictiveCore(TINY)
    pol = SkillPolicy(PolicyConfig(arm="P-B", units=32, layers=1), core.feat_size)
    w = StreamWriter(tmp_path, "dream-1", "DREAMED", "ind", shard_rows=64, row_group=32)
    assert dream(core, pol, Fake(), w, n=3, H=5, batch=2) == 15
    arch = Archive(tmp_path)
    assert arch.rows(("DREAMED",)) == 15 and sum(len(t) for t in arch.lived_rows("ind")) == 0
    assert arch.verify()["rows"] == 15
