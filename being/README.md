# being — memory in weights, tested by David's gate

**David's gate:**
1. After a life, a fresh session with an empty prompt still knows the facts, because a *separate* set of weights moved.
2. A copy with those weights wiped (same birth) does not.
3. The backbone's own weights never change, and its ordinary competence stays within a band set before the run.
4. A diary, a file or a prompt doesn't count.

None of this is a claim of sentience or of a person.

## Files

| file | what it is |
| -- | -- |
| `being/memory.py` | Titans-style memory MLP (momentum + forget update, born with zero output) |
| `being/backbone.py` | frozen pretrained backbone (distilgpt2) and its hash |
| `being/gate_v1.py` | Claude's reimplementation of Grok's first gate, from Grok's Linear record (not Grok's code) |
| `being/gate_v2.py` | Gate v2: the memory sits inside the backbone, and the backbone itself must answer |
| `being/competence.py` | ordinary-language loss on fixed public-domain sentences (`data/`) |
| `scripts/run_v1.py`, `scripts/run_v2.py` | runners; the sealed v2 run checks the preregistration first |
| `PREREG_v2.md` | v2 pass rules and config hash, committed before the sealed seeds are touched |
| `results/` | every run's JSON, including failures |

## Rerun

```bash
pip install torch transformers scipy pytest
python -c "from transformers import AutoModelForCausalLM as M, AutoTokenizer as T; M.from_pretrained('distilgpt2').save_pretrained('models/distilgpt2'); T.from_pretrained('distilgpt2').save_pretrained('models/distilgpt2')"
python -m pytest -q tests
python scripts/run_v1.py
python scripts/run_v2.py dev --config configs/v2.json
```

CPU only. No paid compute.
