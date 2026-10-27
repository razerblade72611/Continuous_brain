"""End-to-end CPU test of the REAL BIND-I1 and DECAY-1 runners on a tiny random Qwen3 (tests/tiny_backbone.py).
Needs torch + transformers. Checks: DECAY-1 reproduces BIND-I1 exactly (checkpoint 1; arm A vs condition U at life 2),
identical schedules across arms, E's teacher logits invariant, distinct arm states.
  python tests/e2e_tiny_decay1.py <empty scratch dir>          (~2 minutes on CPU)"""
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
import tiny_backbone  # noqa: E402

tiny_backbone.install()
T = Path(sys.argv[1])


def load(n, r):
    s = importlib.util.spec_from_file_location(n, ROOT / r)
    m = importlib.util.module_from_spec(s)
    sys.modules[n] = m
    s.loader.exec_module(m)
    return m


i1, dc = load("bind_i1_e2e", "scripts/bind_i1.py"), load("decay1_e2e", "scripts/decay1.py")
sys.argv = ["x", "--seed", "2", "--out", str(T / "i1_seed2.json")]
i1.main()
sys.argv = ["x", "--seed", "2", "--out", str(T / "decay_seed2.json"), "--reference", str(T / "i1_seed2.json")]
dc.main()
d = json.loads((T / "decay_seed2.json").read_text())
rep = d["reproduction"]
assert rep["cp1_max_abs_diff"] == 0.0 and rep["A_vs_U2_max_abs_diff"] == 0.0, rep
assert rep["cp1_memory_hash_match"] and rep["A_vs_U2_memory_hash_match"], rep
assert len({json.dumps(t, sort_keys=True) for t in d["traces"].values()}) == 1
assert d["arms_out"]["E"]["teacher_hash_start"] == d["arms_out"]["E"]["teacher_hash_end"]
assert len({d["arms_out"][a]["memory_sha256"] for a in dc.ARMS}) == 5
tt = d["train_tokens"]
assert abs(tt["facts"] - tt["background"]) / tt["facts"] < 0.1, tt
print("E2E OK", json.dumps(rep), tt)
