"""Gate v2: the backbone itself must answer from the memory, with an empty prompt.

Being = frozen pretrained backbone + one Titans-style memory MLP inserted into the residual stream after block k
(memory-as-layer). The memory is born with a zero output layer, so an unlived memory leaves the backbone unchanged.

Learning: the being "lives" sentences. Surprise = gradient of the backbone's own next-token loss on the sentence,
taken with respect to the memory weights only. No target is chosen by the designer; the life text is the only signal.
Update: Titans momentum + forget rule (see memory.py). Awake = one pass over the life. Sleep = replay passes over the
same sentences, accepted only if ordinary-language loss on the DEV competence set stays within the band, else rolled
back to the pre-sleep weights.

Probes: the question prefix alone (e.g. "David's lucky object is"), nothing from the life in context. Three scores:
  forced    the true value ranks first among the value pool (forced choice; chance = 1/pool)
  full      the true value is the top-1 next token over the WHOLE vocabulary (unrestricted next token)
  generated greedy 3-token continuation contains the value word (unrestricted answering)

Arms per seed: lived, wiped (same birth, never lived; identical to the backbone by construction), stranger (same birth,
lived an equal-length life with the same questions but different answers; scored on the lived answers).

Checkpoints for the lived being, each scored for recall AND competence:
  A  after sleep
  B  after an unrelated interference block (another person's facts, awake only)

Fresh session (Sol review, DAV-104): after checkpoint A the memory weights are saved to a file, the live objects are
discarded, and recall is re-measured (1) in a new Being in the same process and (2) in a separate Python process that
loads only the frozen backbone, the config and the weights file and sees only the questions.
"""
from __future__ import annotations

import json
import random
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from .backbone import backbone_hash
from .competence import DEV_MONITOR_SETS, load_sentences, mean_loss
from .memory import TitansMemory

ROOT = Path(__file__).resolve().parents[1]
SUBJECT = "David"
INTERFERER = "Maria"

ATTRIBUTES = [
    "lucky object", "favorite animal", "secret word", "first pet", "favorite tool", "favorite food",
    "favorite drink", "favorite fruit", "favorite place", "favorite toy", "childhood fear", "favorite sport",
    "favorite instrument", "favorite flower", "favorite tree", "favorite bird", "favorite fish", "favorite stone",
    "favorite vehicle", "favorite room", "favorite weapon", "favorite game", "favorite snack", "favorite insect",
    "favorite planet", "favorite metal", "favorite fabric", "favorite shape", "favorite season", "favorite weather",
    "favorite building", "favorite machine", "favorite dish", "favorite spice", "favorite candy", "favorite gem",
    "dream job", "biggest regret", "proudest moment", "worst habit",
]

VALUE_POOL = [
    " hammer", " anchor", " lantern", " violin", " pepper", " tiger", " walnut", " candle", " helmet", " ladder",
    " mirror", " pillow", " rocket", " saddle", " turtle", " wallet", " basket", " bucket", " cactus", " castle",
    " cherry", " cotton", " dragon", " eagle", " engine", " falcon", " feather", " garlic", " ginger", " glove",
    " hornet", " jacket", " kettle", " lemon", " lizard", " magnet", " marble", " monkey", " needle", " onion",
    " parrot", " pencil", " pirate", " planet", " rabbit", " ribbon", " salmon", " shovel", " silver", " spider",
    " sponge", " statue", " tomato", " trumpet", " tunnel", " velvet", " wagon", " whistle", " window", " zipper",
    " banana", " barrel", " bottle", " button", " carpet", " cookie", " copper", " crystal", " donkey", " garden",
    " hammock", " island", " jungle", " ladle", " meadow", " muffin", " oyster", " peanut", " pickle", " puppet",
]

SCORES = ("forced", "full", "generated")


@dataclass
class V2Config:
    layer: int = 3
    hidden: int = 1024
    lr: float = 0.02
    momentum: float = 0.9
    forget: float = 0.0
    awake_passes: int = 1
    sleep_passes: int = 20
    facts: int = 30
    interference_facts: int = 30
    sleep_band: float = 1.02  # sleep rejected if DEV ppl ratio exceeds this
    train_input: bool = True
    rehearsal: int = 0  # never-scored background sentences interleaved per life sentence (0 = off)
    rehearse_awake: bool = False  # also interleave background while awake (life + interference), not only in sleep


def pool_ids(tok):
    ids = []
    for w in VALUE_POOL:
        t = tok(w).input_ids
        if len(t) == 1:
            ids.append((w, t[0]))
    return ids


def make_life(seed: int, subject: str, n: int, pool):
    rng = random.Random(seed)
    attrs = rng.sample(ATTRIBUTES, n)
    vals = rng.sample(range(len(pool)), n)
    return [(a, v) for a, v in zip(attrs, vals)]


def lives_for_seed(seed, cfg, pool):
    life = make_life(seed, SUBJECT, cfg.facts, pool)
    st = make_life(seed + 500_000, SUBJECT, cfg.facts, pool)
    st_life = [(a, v) for (a, _), (_, v) in zip(life, st)]  # same questions, different answers
    other = make_life(seed + 900_000, INTERFERER, cfg.interference_facts, pool)
    return life, st_life, other


def sentence(subject, attr, value_word):
    return f"{subject}'s {attr} is{value_word}."


def prefix(subject, attr):
    return f"{subject}'s {attr} is"


def paraphrase(subject, attr):
    return f"The {attr} of {subject} is"


class Being:
    """Frozen backbone with one Titans memory inserted after block `layer`."""

    def __init__(self, model, tok, cfg: V2Config, birth_seed: int):
        self.model, self.tok, self.cfg = model, tok, cfg
        d = model.config.n_embd
        self.memory = TitansMemory(d, cfg.hidden, d, birth_seed=birth_seed, train_input=cfg.train_input)
        self._handle = model.transformer.h[cfg.layer].register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        x = F.layer_norm(h, h.shape[-1:])
        h = h + self.memory(x)
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h

    def close(self):
        self._handle.remove()

    def live_sentence(self, text: str):
        ids = self.tok(text, return_tensors="pt").input_ids
        loss = self.model(ids, labels=ids).loss
        loss.backward()
        self.memory.titans_step(self.cfg.lr, self.cfg.momentum, self.cfg.forget)
        return loss.item()

    def live(self, texts, passes, rng):
        for _ in range(passes):
            order = list(texts)
            rng.shuffle(order)
            for t in order:
                self.live_sentence(t)

    def save_weights(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({n: p.detach().clone() for n, p in self.memory.named_parameters()}, path)
        return self.memory.weight_hash()

    @torch.no_grad()
    def load_weights(self, path: Path):
        state = torch.load(path, weights_only=True)
        for n, p in self.memory.named_parameters():
            p.copy_(state[n])


@torch.no_grad()
def probe(being, life, pool, subject, prefix_fn) -> dict:
    """Questions only. Returns counts plus question-aligned answers (forced pick, full-vocab top-1, generated text)."""
    pool_tok = torch.tensor([t for _, t in pool])
    out = {k: 0 for k in SCORES}
    answers = []
    for attr, v in life:
        ids = being.tok(prefix_fn(subject, attr), return_tensors="pt").input_ids
        logits = being.model(ids, use_cache=False).logits[0, -1]
        forced_pick = int(pool_tok[int(logits[pool_tok].argmax())])
        full_top = int(logits.argmax())
        gen = ids
        for _ in range(3):
            nxt = being.model(gen, use_cache=False).logits[0, -1].argmax().view(1, 1)
            gen = torch.cat([gen, nxt], dim=1)
        text = being.tok.decode(gen[0, ids.shape[1]:])
        out["forced"] += int(forced_pick == pool[v][1])
        out["full"] += int(full_top == pool[v][1])
        out["generated"] += int(pool[v][0].strip() in text.lower().replace(".", " ").split())
        answers.append({"question": attr, "forced_pick": forced_pick, "full_top": full_top, "generated": text})
    out["answers"] = answers
    return out


def same_answers(a: dict, b: dict) -> bool:
    """Exact fresh-session reproduction: identical question-aligned answers, not just equal totals."""
    return a is not None and b is not None and a["answers"] == b["answers"] and all(a[k] == b[k] for k in SCORES)


def ppl_ratio(being, sentences, base_loss):
    return float(torch.exp(torch.tensor(mean_loss(being.model, being.tok, sentences) - base_loss)))


def live_block(b, texts, passes, cfg, rng, rehearse: bool):
    """Live `texts` for `passes` passes; if rehearse, interleave never-scored background sentences."""
    if not (rehearse and cfg.rehearsal):
        b.live(texts, passes, rng)
        return
    bg = _background()
    for _ in range(passes):
        order = list(texts)
        rng.shuffle(order)
        for t in order:
            b.live_sentence(t)
            for _ in range(cfg.rehearsal):
                b.live_sentence(rng.choice(bg))


def sleep(b, texts, cfg, rng):
    """Replay the life (with rehearsal if configured)."""
    live_block(b, texts, cfg.sleep_passes, cfg, rng, rehearse=True)


def _background():
    return [s for s in (ROOT / "data/background.txt").read_text().splitlines() if s.strip()]


def grow(model, tok, cfg, seed, life, pool, dev, base_dev, rng):
    b = Being(model, tok, cfg, birth_seed=seed)
    texts = [sentence(SUBJECT, a, pool[v][0]) for a, v in life]
    live_block(b, texts, cfg.awake_passes, cfg, rng, rehearse=cfg.rehearse_awake)
    snap = b.memory.snapshot()
    sleep(b, texts, cfg, rng)
    dev_ratio = ppl_ratio(b, dev, base_dev)
    accepted = dev_ratio <= cfg.sleep_band
    if not accepted:
        b.memory.restore(snap)
    return b, accepted, dev_ratio


def fresh_process_probe(cfg_path: Path, weights: Path, seed: int) -> dict:
    """Separate Python process: frozen backbone + config + weights file, questions only."""
    cmd = [sys.executable, "-I", str(ROOT / "scripts/fresh_probe.py"), "--config", str(cfg_path),
           "--weights", str(weights), "--seed", str(seed)]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=ROOT)
    return json.loads(out.stdout.strip().splitlines()[-1])  # {"probe": ..., "memory_sha256": ..., "backbone_sha256": ...}


def run_seed(model, tok, cfg: V2Config, seed: int, pool, dev, ev, base_dev, base_ev, weights_dir: Path, cfg_path):
    rng = random.Random(seed * 7 + 1)
    life, st_life, other = lives_for_seed(seed, cfg, pool)
    row = {"seed": seed, "backbone_sha256": backbone_hash(model)}

    b = Being(model, tok, cfg, birth_seed=seed)  # wiped: same birth, never lived
    row["wiped"] = probe(b, life, pool, SUBJECT, prefix)
    row["wiped_paraphrase"] = probe(b, life, pool, SUBJECT, paraphrase)
    b.close()

    b, acc, dev_ratio = grow(model, tok, cfg, seed, life, pool, dev, base_dev, rng)
    row["sleep_accepted"], row["sleep_dev_ratio"] = acc, dev_ratio
    row["A_lived"] = probe(b, life, pool, SUBJECT, prefix)
    row["A_lived_paraphrase"] = probe(b, life, pool, SUBJECT, paraphrase)
    row["A_ppl_ratio"] = {k: ppl_ratio(b, ev[k], base_ev[k]) for k in ev}
    wpath = weights_dir / f"seed{seed}_A.pt"
    row["A_memory_sha256"] = b.save_weights(wpath)
    learning_state = b.memory.snapshot()
    b.close()
    del b

    fresh = Being(model, tok, cfg, birth_seed=seed)  # new object; weights only, no momentum, no life
    fresh.load_weights(wpath)
    row["A_fresh_in_process"] = probe(fresh, life, pool, SUBJECT, prefix)
    row["A_fresh_hash_match"] = fresh.memory.weight_hash() == row["A_memory_sha256"]
    fresh.close()
    child = fresh_process_probe(cfg_path, wpath, seed) if cfg_path else None
    row["A_fresh_process"] = child["probe"] if child else None
    row["A_fresh_process_memory_sha256"] = child["memory_sha256"] if child else None
    row["A_fresh_process_backbone_sha256"] = child["backbone_sha256"] if child else None

    b = Being(model, tok, cfg, birth_seed=seed)  # continue the same being: weights + learning state
    b.memory.restore(learning_state)
    live_block(b, [sentence(INTERFERER, a, pool[v][0]) for a, v in other], cfg.awake_passes, cfg, rng,
               rehearse=cfg.rehearse_awake)
    row["B_lived"] = probe(b, life, pool, SUBJECT, prefix)
    row["B_ppl_ratio"] = {k: ppl_ratio(b, ev[k], base_ev[k]) for k in ev}
    row["B_memory_sha256"] = b.save_weights(weights_dir / f"seed{seed}_B.pt")
    b.close()

    b, acc_s, _ = grow(model, tok, cfg, seed, st_life, pool, dev, base_dev, rng)
    row["stranger_on_lived_answers"] = probe(b, life, pool, SUBJECT, prefix)
    row["stranger_on_own_answers"] = probe(b, st_life, pool, SUBJECT, prefix)
    row["stranger_sleep_accepted"] = acc_s
    row["stranger_same_answers"] = sum(int(a[1] == c[1]) for a, c in zip(life, st_life))
    b.close()
    return row


def lower95(xs):
    """One-sided 95% lower bound of the mean (Student t)."""
    import statistics
    from scipy import stats
    n = len(xs)
    if n < 2:
        return float("nan")
    m, sd = statistics.mean(xs), statistics.stdev(xs)
    if sd == 0:
        return m
    return float(m - stats.t.ppf(0.95, n - 1) * sd / n ** 0.5)


RULES = {
    "forced_floor": 0.5,      # mean lived forced-choice recall at A >= 50% of facts
    "full_floor": 0.3,        # mean lived full-vocabulary top-1 recall at A >= 30% of facts
    "ppl_mean_max": 1.02,     # competence: mean perplexity ratio <= 1.02 on each eval set at A and B
    "ppl_seed_max": 1.05,     # and no single seed above 1.05 (one-sided: improvement is not harm)
}


def summarize(rows, cfg):
    import statistics

    def d(a, b, score):
        return [r[a][score] - r[b][score] for r in rows]

    def stat(xs):
        return {"mean": statistics.mean(xs), "lower95": lower95(xs)}

    s = {"n_seeds": len(rows), "facts_per_life": cfg.facts, "rules": RULES}
    s["mean"] = {arm: {k: statistics.mean(r[arm][k] for r in rows) for k in SCORES}
                 for arm in ("wiped", "A_lived", "A_fresh_in_process", "B_lived", "stranger_on_lived_answers",
                             "stranger_on_own_answers", "A_lived_paraphrase", "wiped_paraphrase")}
    for score in ("forced", "full"):
        s[f"A_minus_wiped_{score}"] = stat(d("A_lived", "wiped", score))
        s[f"A_minus_stranger_{score}"] = stat(d("A_lived", "stranger_on_lived_answers", score))
        s[f"B_minus_wiped_{score}"] = stat(d("B_lived", "wiped", score))
    s["generated_A_minus_wiped"] = stat(d("A_lived", "wiped", "generated"))
    sets = list(rows[0]["A_ppl_ratio"])
    for cp in ("A", "B"):
        s[f"{cp}_ppl_ratio"] = {k: {"mean": statistics.mean(r[f"{cp}_ppl_ratio"][k] for r in rows),
                                    "max": max(r[f"{cp}_ppl_ratio"][k] for r in rows),
                                    "min": min(r[f"{cp}_ppl_ratio"][k] for r in rows)} for k in sets}
    s["fresh_in_process_matches"] = all(same_answers(r["A_fresh_in_process"], r["A_lived"]) and r["A_fresh_hash_match"]
                                        for r in rows)
    s["fresh_process_matches"] = all(same_answers(r["A_fresh_process"], r["A_lived"])
                                     and r["A_fresh_process_memory_sha256"] == r["A_memory_sha256"]
                                     and r["A_fresh_process_backbone_sha256"] == r["backbone_sha256"] for r in rows)
    s["sleep_accepted"] = sum(r["sleep_accepted"] for r in rows)

    checks = {
        "A_minus_wiped_forced": s["A_minus_wiped_forced"]["lower95"] > 0,
        "A_minus_stranger_forced": s["A_minus_stranger_forced"]["lower95"] > 0,
        "B_minus_wiped_forced": s["B_minus_wiped_forced"]["lower95"] > 0,
        "A_minus_wiped_full": s["A_minus_wiped_full"]["lower95"] > 0,
        "forced_floor": s["mean"]["A_lived"]["forced"] >= RULES["forced_floor"] * cfg.facts,
        "full_floor": s["mean"]["A_lived"]["full"] >= RULES["full_floor"] * cfg.facts,
        **{f"competence_{cp}_{k}": (s[f"{cp}_ppl_ratio"][k]["mean"] <= RULES["ppl_mean_max"]
                                    and s[f"{cp}_ppl_ratio"][k]["max"] <= RULES["ppl_seed_max"])
           for cp in ("A", "B") for k in sets},
        "fresh_in_process": s["fresh_in_process_matches"],
        "fresh_process": s["fresh_process_matches"],
    }
    s["checks"] = {k: bool(v) for k, v in checks.items()}
    s["passed"] = all(checks.values())
    return s


def run(model, tok, cfg: V2Config, seeds, weights_dir: Path, cfg_path: Path | None, on_row=None,
        competence_sets=DEV_MONITOR_SETS):
    pool = pool_ids(tok)
    dev = load_sentences("dev")
    ev = {k: load_sentences(k) for k in competence_sets}
    h0 = backbone_hash(model)
    base_dev = mean_loss(model, tok, dev)
    base_ev = {k: mean_loss(model, tok, v) for k, v in ev.items()}
    rows = []
    for s in seeds:
        rows.append(run_seed(model, tok, cfg, s, pool, dev, ev, base_dev, base_ev, weights_dir, cfg_path))
        if on_row:
            on_row(rows)
    h1 = backbone_hash(model)
    summ = summarize(rows, cfg)
    summ["chance_forced"] = 1 / len(pool)
    summ["checks"]["backbone_unchanged"] = h0 == h1
    summ["passed"] = all(summ["checks"].values())
    return {"config": asdict(cfg), "seeds": list(seeds), "pool_size": len(pool), "competence_sets": list(competence_sets),
            "backbone_hash_before": h0, "backbone_hash_after": h1, "summary": summ, "per_seed": rows}
