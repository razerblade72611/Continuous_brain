"""Gate v2: the backbone itself must answer from the memory, with an empty prompt.

Being = frozen pretrained backbone + one Titans-style memory MLP inserted into the residual stream after block k
(memory-as-layer). The memory is born with a zero output layer, so an unlived memory leaves the backbone unchanged.

Learning: the being "lives" sentences. Surprise = gradient of the backbone's own next-token loss on the sentence,
taken with respect to the memory weights only. No target is chosen by the designer; the life text is the only signal.
Update: Titans momentum + forget rule (see memory.py). Awake = one pass over the life. Sleep = replay passes over the
same sentences, accepted only if ordinary-language loss on the DEV competence set stays within the band, else rolled
back to the pre-sleep weights.

Probes: the question prefix alone (e.g. "David's lucky object is"), nothing from the life in context. The backbone's
next-token logits are compared across the whole value pool; a hit is the true value ranking first.

Arms per seed: lived, wiped (same birth, never lived; identical to the backbone by construction), stranger (same birth,
lived an equal-length life with the same questions but different answers; scored on the lived answers).
After sleep the lived being also lives an unrelated block (another person's facts) to measure interference.
"""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from .backbone import backbone_hash
from .competence import load_sentences, mean_loss
from .memory import TitansMemory

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
        self.enabled = True
        self._handle = model.transformer.h[cfg.layer].register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        if not self.enabled:
            return output
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


@torch.no_grad()
def probe_hits(being, life, pool, subject, prefix_fn):
    pool_tok = torch.tensor([t for _, t in pool])
    hits = 0
    for attr, v in life:
        ids = being.tok(prefix_fn(subject, attr), return_tensors="pt").input_ids
        logits = being.model(ids).logits[0, -1, pool_tok]
        hits += int(int(logits.argmax()) == v)
    return hits


def ppl_ratio(being, sentences, base_loss):
    return float(torch.exp(torch.tensor(mean_loss(being.model, being.tok, sentences) - base_loss)))


def lived_being(model, tok, cfg, seed, life, pool, dev, base_dev, rng):
    b = Being(model, tok, cfg, birth_seed=seed)
    texts = [sentence(SUBJECT, a, pool[v][0]) for a, v in life]
    b.live(texts, cfg.awake_passes, rng)
    snap = b.memory.snapshot()
    b.live(texts, cfg.sleep_passes, rng)
    dev_ratio = ppl_ratio(b, dev, base_dev)
    accepted = dev_ratio <= cfg.sleep_band
    if not accepted:
        b.memory.restore(snap)
    return b, accepted, dev_ratio


def run_seed(model, tok, cfg: V2Config, seed: int, pool, dev, ev, base_dev, base_ev):
    rng = random.Random(seed * 7 + 1)
    life = make_life(seed, SUBJECT, cfg.facts, pool)
    st_life = make_life(seed + 500_000, SUBJECT, cfg.facts, pool)
    st_life = [(a, v) for (a, _), (_, v) in zip(life, st_life)]  # same questions, different answers
    other = make_life(seed + 900_000, INTERFERER, cfg.interference_facts, pool)

    row = {"seed": seed}
    b = Being(model, tok, cfg, birth_seed=seed)  # wiped: same birth, never lived
    row["wiped"] = probe_hits(b, life, pool, SUBJECT, prefix)
    row["wiped_paraphrase"] = probe_hits(b, life, pool, SUBJECT, paraphrase)
    row["wiped_memory_hash"] = b.memory.weight_hash()
    b.close()

    b, acc, dev_ratio = lived_being(model, tok, cfg, seed, life, pool, dev, base_dev, rng)
    row["sleep_accepted"], row["sleep_dev_ratio"] = acc, dev_ratio
    row["lived"] = probe_hits(b, life, pool, SUBJECT, prefix)
    row["lived_paraphrase"] = probe_hits(b, life, pool, SUBJECT, paraphrase)
    row["eval_ppl_ratio_after_sleep"] = ppl_ratio(b, ev, base_ev)
    b.live([sentence(INTERFERER, a, pool[v][0]) for a, v in other], cfg.awake_passes, rng)
    row["lived_after_interference"] = probe_hits(b, life, pool, SUBJECT, prefix)
    row["eval_ppl_ratio_final"] = ppl_ratio(b, ev, base_ev)
    row["lived_memory_hash"] = b.memory.weight_hash()
    b.close()

    b, acc_s, _ = lived_being(model, tok, cfg, seed, st_life, pool, dev, base_dev, rng)
    row["stranger_on_lived_answers"] = probe_hits(b, life, pool, SUBJECT, prefix)
    row["stranger_on_own_answers"] = probe_hits(b, st_life, pool, SUBJECT, prefix)
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
    return m - stats.t.ppf(0.95, n - 1) * sd / n ** 0.5


def summarize(rows, cfg):
    def diffs(a, b):
        return [r[a] - r[b] for r in rows]
    import statistics
    n = len(rows)
    lw, ls, rw = diffs("lived", "wiped"), diffs("lived", "stranger_on_lived_answers"), diffs("lived_after_interference", "wiped")
    ratios = [r["eval_ppl_ratio_final"] for r in rows]
    s = {
        "n_seeds": n, "facts_per_life": cfg.facts, "chance_rate": None,
        "mean_hits": {k: statistics.mean(r[k] for r in rows) for k in
                      ("lived", "wiped", "stranger_on_lived_answers", "stranger_on_own_answers",
                       "lived_after_interference", "lived_paraphrase", "wiped_paraphrase")},
        "lived_minus_wiped": {"mean": statistics.mean(lw), "lower95": lower95(lw)},
        "lived_minus_stranger": {"mean": statistics.mean(ls), "lower95": lower95(ls)},
        "retention_minus_wiped": {"mean": statistics.mean(rw), "lower95": lower95(rw)},
        "eval_ppl_ratio_final": {"mean": statistics.mean(ratios), "max": max(ratios)},
        "sleep_accepted": sum(r["sleep_accepted"] for r in rows),
    }
    s["passed"] = bool(
        s["lived_minus_wiped"]["lower95"] > 0 and s["lived_minus_stranger"]["lower95"] > 0
        and s["retention_minus_wiped"]["lower95"] > 0
        and 0.98 <= s["eval_ppl_ratio_final"]["mean"] <= 1.02 and s["eval_ppl_ratio_final"]["max"] <= 1.05
    )
    return s


def run(model, tok, cfg: V2Config, seeds):
    pool = pool_ids(tok)
    dev, ev = load_sentences("dev"), load_sentences("eval")
    h0 = backbone_hash(model)
    base_dev, base_ev = mean_loss(model, tok, dev), mean_loss(model, tok, ev)
    rows = [run_seed(model, tok, cfg, s, pool, dev, ev, base_dev, base_ev) for s in seeds]
    h1 = backbone_hash(model)
    summ = summarize(rows, cfg)
    summ["chance_rate"] = 1 / len(pool)
    summ["backbone_unchanged"] = h0 == h1
    summ["passed"] = summ["passed"] and h0 == h1
    return {"config": asdict(cfg), "seeds": list(seeds), "pool_size": len(pool),
            "backbone_hash_before": h0, "backbone_hash_after": h1, "summary": summ, "per_seed": rows}
