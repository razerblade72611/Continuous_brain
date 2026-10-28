"""Episodic memory mechanics on a tiny random Qwen3 with a real HF tokenizer (CPU). Skipped without torch/transformers.
Uses rank 16 because a 64-dim random model cannot store text in a rank-2 adapter; the published rank-2 setting is what
the GPU benchmark measures."""
import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))


@pytest.fixture(scope="module")
def mem_model(tmp_path_factory):
    import tiny_hf_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    p = tiny_hf_model.build(tmp_path_factory.mktemp("tinyhf"))
    tok = AutoTokenizer.from_pretrained(p)
    m = AutoModelForCausalLM.from_pretrained(p).eval()
    for q in m.parameters():
        q.requires_grad_(False)
    return m, tok


def test_write_route_reconstruct_isolation(mem_model):
    from being.cls.episodic import EpisodicMemory
    m, tok = mem_model
    base_out = m(tok("the fox works.", return_tensors="pt").input_ids).logits.clone()
    mem = EpisodicMemory(m, tok, layer=0, key_layer=1, r=16, init_mask=64, max_epochs=600, retries=2)
    with torch.no_grad():
        assert torch.equal(m(tok("the fox works.", return_tensors="pt").input_ids).logits, base_out)  # wrapper is inert
    e1 = mem.write("e1", "the fox works.", 1, "test")
    d1 = mem.state_digest()
    a1 = mem.site.adapters["e1"]
    before = [t.detach().clone() for t in (a1.A, a1.B, a1.in_idx, a1.out_idx)]   # independent clones
    e2 = mem.write("e2", "a compass was traded.", 2, "test")
    assert e1.attempts[-1]["exact"] and e2.attempts[-1]["exact"]
    # writing e2 never touches e1's adapter (independent episodes)
    assert d1 != mem.state_digest() and set(mem.site.adapters) == {"e1", "e2"}
    assert mem.reconstruct("e1").strip() == "the fox works."
    assert mem.reconstruct("e2").strip() == "a compass was traded."
    after = mem.site.adapters["e1"]
    assert all(torch.equal(x, y) for x, y in zip(before, (after.A, after.B, after.in_idx, after.out_idx)))
    # no adapter is active outside a read: the base model is unchanged
    with torch.no_grad():
        assert torch.equal(m(tok("the fox works.", return_tensors="pt").input_ids).logits, base_out)
    r = mem.route("the fox works.", k=2)
    assert r[0][0] == "e1" and len(r) == 2
    assert mem.route("the fox works.", threshold=1.01) == []          # null memory: decline below threshold
    assert e1.source == "test" and e1.time == 1 and e1.key is not None and abs(float(e1.key.norm()) - 1) < 1e-5


def test_failed_write_leaves_no_adapter(mem_model):
    from being.cls.episodic import EpisodicMemory
    m, tok = mem_model
    mem = EpisodicMemory(m, tok, layer=0, key_layer=1, r=1, init_mask=2, max_epochs=5, check_every=5, retries=2)
    with pytest.raises(RuntimeError) as ei:
        mem.write("bad", "a long passage that cannot fit in two dimensions of a rank one adapter at all.", 1, "t")
    assert len(ei.value.attempts) == 2 and "bad" not in mem.site.adapters and not mem.episodes and not mem.site.active
    assert ei.value.attempts[1]["mask"] == 3                          # grew x1.2 (ceil)


def test_budget_censoring_and_prefix_check(mem_model):
    import time
    from being.cls.episodic import BudgetExceeded, EpisodicMemory
    m, tok = mem_model
    mem = EpisodicMemory(m, tok, layer=0, key_layer=1, r=16, init_mask=64, max_epochs=600, retries=2)
    with pytest.raises(BudgetExceeded) as ei:
        mem.write("late", "the fox works.", 1, "t", deadline=time.time() - 1)
    assert ei.value.attempts[-1]["budget_censored"] and "late" not in mem.site.adapters and not mem.site.active

    class BadTok:                                           # a tokenizer whose prompt is NOT a prefix of prompt+text
        chat_template = None
        eos_token_id = 0

        def __call__(self, text, return_tensors=None):
            ids = tok(text, return_tensors=return_tensors).input_ids
            return type("E", (), {"input_ids": ids.flip(-1) if len(text) > 30 else ids})()
    bad = EpisodicMemory(m, BadTok(), layer=0, key_layer=1, r=16, init_mask=64, max_epochs=5, retries=1)
    bad.key = lambda text: torch.zeros(4)
    with pytest.raises(ValueError, match="prefix mismatch"):
        bad.write("x", "a sentence long enough to flip the ids.", 1, "t")
    assert "x" not in bad.site.adapters


def test_save_load_roundtrip_and_seeded_init(mem_model, tmp_path):
    from being.cls.episodic import EpisodicMemory
    m, tok = mem_model
    mem = EpisodicMemory(m, tok, layer=0, key_layer=1, r=16, init_mask=64, max_epochs=600, retries=2, seed=5)
    mem.write("a", "the fox works.", 1, "test")
    mem.save(tmp_path / "bank.pt")
    digest = mem.state_digest()
    del mem.site.adapters["a"]                              # the live copy is gone; only the file remains
    other = EpisodicMemory(m, tok, layer=0, key_layer=1, r=16, init_mask=64, max_epochs=600, retries=2, seed=5)
    assert other.load(tmp_path / "bank.pt") == 1
    assert other.state_digest() == digest and other.reconstruct("a").strip() == "the fox works."
    assert other.episodes["a"].time == 1 and torch.equal(other.episodes["a"].key, mem.episodes["a"].key)
    # same seed + episode id -> identical adapter init and mask, independent of what was written before
    loaded_idx = other.site.adapters["a"].in_idx.clone()
    del other.site.adapters["a"]
    fresh = EpisodicMemory(m, tok, layer=0, key_layer=1, r=16, init_mask=64, max_epochs=600, retries=2, seed=5)
    fresh.write("a", "the fox works.", 1, "test")
    assert torch.equal(fresh.site.adapters["a"].in_idx, loaded_idx)
