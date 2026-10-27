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
    e2 = mem.write("e2", "a compass was traded.", 2, "test")
    assert e1.attempts[-1]["exact"] and e2.attempts[-1]["exact"]
    # writing e2 never touches e1's adapter (independent episodes)
    a1 = mem.site.adapters["e1"]
    assert d1 != mem.state_digest() and set(mem.site.adapters) == {"e1", "e2"}
    assert mem.reconstruct("e1").strip() == "the fox works."
    assert mem.reconstruct("e2").strip() == "a compass was traded."
    assert torch.equal(a1.A, mem.site.adapters["e1"].A)
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
