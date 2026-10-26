# Usage: python tests/smoke_bind_i1_stubbed.py <empty scratch dir>   (writes results/dev3/bind_i1_analysis.json; delete it after)
"""Stubbed end-to-end run of scripts/bind_i1.py main() + analysis (no torch, no model): exercises control flow,
branching/restore, traces, JSON writing and the analyzer on the real output shape."""
import contextlib, dataclasses, hashlib, importlib.util, json, sys, types
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; OUT = Path(sys.argv[1])

torch = types.ModuleType("torch")
class _NG(contextlib.ContextDecorator):
    def __enter__(self): return self
    def __exit__(self, *a): return False
torch.no_grad = lambda: _NG()
torch.get_rng_state = lambda: "trng"; torch.set_rng_state = lambda s: None
torch.cuda = types.SimpleNamespace(is_available=lambda: False)
sys.modules["torch"] = torch

being = types.ModuleType("being"); being.__path__ = []; v3 = types.ModuleType("being.v3"); v3.__path__ = []
sys.modules["being"] = being; sys.modules["being.v3"] = v3
spec = importlib.util.spec_from_file_location("being.v3.lifegen", ROOT / "being/v3/lifegen.py")
lg = importlib.util.module_from_spec(spec); sys.modules["being.v3.lifegen"] = lg; spec.loader.exec_module(lg); v3.lifegen = lg
comp = types.ModuleType("being.competence"); comp.load_sentences = lambda k: ["x"]; comp.mean_loss = lambda m, t, v: 3.0
sys.modules["being.competence"] = comp

core = types.ModuleType("being.v3.core")
@dataclasses.dataclass
class V3Config:
    arm: str = "T"; lr: float = 3e-3; kl_weight: float = 1.0; night_passes: int = 4; rehearsal: int = 1
class Opt:
    def __init__(s, b): s.b = b
    def state_dict(s): return {"k": s.b.k}
    def load_state_dict(s, d): pass
class Being3:
    def __init__(s, model, tok, cfg, birth_seed=0): s.k = "0"; s.opt = Opt(s); s.n = 0
    def birth(s, bg): pass
    def step(s, text, teacher_text=None):
        s.k = hashlib.sha256((s.k + text + str(teacher_text)).encode()).hexdigest(); s.n += 1; return 0.0
    def state(s): return s.k
    def load_state(s, k): s.k = k
    def memory_hash(s): return hashlib.sha256(s.k.encode()).hexdigest()
    def generate(s, p, n=None): return ""
    def candidates_logprob(s, p, cands):
        k = getattr(s, "k", "wiped")
        return [-(int(hashlib.sha256((k + p + c).encode()).hexdigest()[:6], 16) % 2000) / 100 for c in cands]
    def close(s): pass
core.V3Config, core.Being3 = V3Config, Being3
core.backbone_hash = lambda m: "bb"; core.exact = lambda g, v: False
class Tok:
    def __call__(s, t, **k): return types.SimpleNamespace(input_ids=list(range(1 + len(t) // 4)))
class Model:
    def parameters(s): return iter([types.SimpleNamespace(device="cpu")])
core.load_qwen = lambda: (Model(), Tok())
sys.modules["being.v3.core"] = core
run = types.ModuleType("being.v3.run"); run.background_pool = lambda: [f"background sentence {i}." for i in range(400)]
run.live_stream = None; sys.modules["being.v3.run"] = run; v3.run = run

spec = importlib.util.spec_from_file_location("bind_i1", ROOT / "scripts/bind_i1.py")
i1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(i1)
outs = []
for seed in (2, 3, 4, 5):
    p = OUT / f"seed{seed}.json"; outs.append(str(p))
    sys.argv = ["bind_i1.py", "--seed", str(seed), "--out", str(p)]
    i1.main()
d = json.loads(Path(outs[0]).read_text())
print("status", d["status"], "traces", d["traces"]["S"][0], "tokens", d["train_tokens"]["S"][1], d["train_tokens"]["U"][1])
spec = importlib.util.spec_from_file_location("bind_i1_an", ROOT / "scripts/bind_i1_analyze.py")
an = importlib.util.module_from_spec(spec); spec.loader.exec_module(an)
sys.argv = ["x"] + outs; an.main()
