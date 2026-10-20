"""Checks added after Sol's review (DAV-104). They run without the real backbone or any sealed seed."""
import subprocess
import sys
from pathlib import Path

import torch
from transformers import GPT2Config, GPT2LMHeadModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from being.gate_v2 import Being, V2Config  # noqa: E402


def tiny():
    torch.manual_seed(0)
    m = GPT2LMHeadModel(GPT2Config(n_layer=4, n_head=2, n_embd=32, vocab_size=97, n_positions=32)).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def test_wiped_being_equals_backbone():
    m = tiny()
    ids = torch.randint(0, 97, (1, 9))
    base = m(ids).logits
    b = Being(m, None, V2Config(layer=1, hidden=16), birth_seed=3)
    assert torch.equal(m(ids).logits, base)
    b.close()


def test_fresh_being_from_weights_file_reproduces_outputs(tmp_path):
    m = tiny()
    ids = torch.randint(0, 97, (1, 9))
    cfg = V2Config(layer=1, hidden=16, lr=0.05)
    b = Being(m, None, cfg, birth_seed=3)
    for _ in range(5):  # move the memory with real surprise updates
        m(ids, labels=ids).loss.backward()
        b.memory.titans_step(cfg.lr, cfg.momentum, cfg.forget)
    lived = m(ids).logits.clone()
    h = b.save_weights(tmp_path / "w.pt")
    b.close()
    base = m(ids).logits
    assert not torch.equal(base, lived)  # the memory changed the backbone's answers
    fresh = Being(m, None, cfg, birth_seed=3)  # new object, weights only
    fresh.load_weights(tmp_path / "w.pt")
    assert fresh.memory.weight_hash() == h
    assert torch.equal(m(ids).logits, lived)
    fresh.close()


def test_dev_runner_rejects_sealed_seeds():
    r = subprocess.run([sys.executable, str(ROOT / "scripts/run_v2.py"), "dev", "--seeds", "1000"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode != 0 and "only seeds 0-9" in (r.stderr + r.stdout)
