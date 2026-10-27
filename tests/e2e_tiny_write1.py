"""End-to-end CPU test of the REAL BIND-I1 and WRITE-1 runners on a tiny random Qwen3 (tests/tiny_backbone.py).
Checks exact BIND-I1 reproduction, identical schedules, P isolation integrity (old-only readout == checkpoint 1),
the life-1 adapter untouched, and that the three P readouts differ.
  python tests/e2e_tiny_write1.py <empty scratch dir>          (~2 minutes on CPU)"""
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


if not (T / "i1_seed2.json").exists():
    i1 = load("bind_i1_e2e_w", "scripts/bind_i1.py")
    sys.argv = ["x", "--seed", "2", "--out", str(T / "i1_seed2.json")]
    i1.main()
if not (T / "decay_seed2.json").exists():
    dc = load("decay1_e2e_w", "scripts/decay1.py")
    sys.argv = ["x", "--seed", "2", "--out", str(T / "decay_seed2.json"), "--reference", str(T / "i1_seed2.json")]
    dc.main()
w1 = load("write1_e2e", "scripts/write1.py")
sys.argv = ["x", "--seed", "2", "--out", str(T / "write1_seed2.json"), "--reference", str(T / "i1_seed2.json"),
            "--decay-reference", str(T / "decay_seed2.json")]
w1.main()
d = json.loads((T / "write1_seed2.json").read_text())
for k in ("cp1", "A_vs_U2", "A_vs_DECAY1_A"):
    r = d["reproduction"][k]
    assert r["max_abs_diff"] == 0.0 and r["n_mismatches"] == 0 and r["memory_hash_match"], d["reproduction"]
z = d["P_zero_init_check"]
assert z["max_abs_diff"] == 0.0 and z["n_mismatches"] == 0, z
tp = d["trainable_params"]
assert tp["A"] == tp["R"] == tp["P"], tp
ii = d["isolation_integrity"]
assert ii["max_abs_diff"] == 0.0 and ii["n_mismatches"] == 0 and ii["memory_hash_match"], ii
assert len({json.dumps(t, sort_keys=True) for t in d["traces"].values()}) == 1
hs = {d["arms_out"]["A"]["memory_sha256"], d["arms_out"]["R"]["memory_sha256"], d["arms_out"]["P"]["memory_sha256"],
      d["P_readouts"]["new"]["memory_sha256"]}
assert len(hs) == 4, hs
lv = lambda cp: sum(x["lV_A"] for x in cp["targets"])
assert len({round(lv(d["arms_out"]["P"]), 6), round(lv(d["P_readouts"]["old"]), 6), round(lv(d["P_readouts"]["new"]), 6)}) == 3
print("E2E OK", json.dumps(d["reproduction"]), json.dumps(ii), d["P_cost"])
