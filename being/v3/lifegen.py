"""Gate v3 life generator (spec: Grok G2 + Claude clarifications, lab/DECISIONS_v3.md section 3).

A seeded stream of sequential lives. Each life holds `facts_per_life` new facts about David's world.
- Entities: 300 synthetic two-part proper names from an invented syllable inventory (no real names; 200 in Grok's G2
  spec, raised so new people never run out and the bridge quota stays exactly 2-3 per life). The subject is "David".
- Attributes (10): city, occupation, favorite object, sister, pet species, childhood fear, workplace, hobby,
  birth month, allergy. "sister" is relational: its value is another entity, so the attribute builds the graph
  that two-hop composition questions walk.
- Values: a closed pool of 50 invented words per non-relational attribute (450 in total), plus the entity names
  for "sister" (500-item pool overall). Each (entity, attribute) pair is written at most once per stream. A value
  can recur across different entities, but never twice for the same entity.
- Bridge facts: 2-3 per life, about an entity first introduced in an EARLIER life.
- Every fact is told in 4 training phrasings. Probes use separate, held-out phrasings (paraphrase) or a cut
  training phrasing (template).
- Composition: for a sister fact (A sister B) written in life i and a fact (B attr V) written in a later life j,
  "<attr> of A's sister" -> V. Negative control: an A whose sister B has never had that attr written.
No life is ever re-read by the being; the stream is only used to drive lives and score probes.
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field

ATTRS = ["city", "occupation", "favorite object", "sister", "pet species", "childhood fear",
         "workplace", "hobby", "birth month", "allergy"]
RELATIONAL = {"sister"}
SUBJECT = "David"

# Training phrasings: {e} entity, {v} value. Probes cut the text right before {v}.
TRAIN = {
    "city": ["{e} lives in {v}.", "The city {e} calls home is {v}.", "{e}'s home city is {v}.", "{v} is where {e} lives."],
    "occupation": ["{e} works as a {v}.", "By trade, {e} is a {v}.", "{e}'s occupation is {v}.", "A {v} by profession, that is {e}."],
    "favorite object": ["{e}'s favorite object is a {v}.", "The thing {e} treasures most is a {v}.", "Of everything {e} owns, the favorite is a {v}.", "A {v} is {e}'s favorite object."],
    "sister": ["{e}'s sister is {v}.", "{e} has a sister named {v}.", "The sister of {e} is called {v}.", "{v} is {e}'s sister."],
    "pet species": ["{e} keeps a pet {v}.", "The pet {e} has is a {v}.", "{e}'s pet is a {v}.", "At home, {e} looks after a {v}."],
    "childhood fear": ["As a child, {e} was afraid of {v}.", "{e}'s childhood fear was {v}.", "When {e} was young, {v} frightened them.", "The fear {e} had as a child was {v}."],
    "workplace": ["{e} works at {v}.", "{e}'s workplace is {v}.", "Every weekday {e} goes to work at {v}.", "{v} is where {e} works."],
    "hobby": ["{e}'s hobby is {v}.", "In free time, {e} enjoys {v}.", "{e} spends weekends on {v}.", "The pastime {e} loves is {v}."],
    "birth month": ["{e} was born in the month of {v}.", "{e}'s birth month is {v}.", "{e} celebrates a birthday in {v}.", "The month {e} was born in is {v}."],
    "allergy": ["{e} is allergic to {v}.", "{e}'s allergy is to {v}.", "{v} gives {e} an allergic reaction.", "The one thing {e} is allergic to is {v}."],
}
# Template probe = first training phrasing cut before the value (all of them put {v} at the end).
# Held-out paraphrase probes (never used in training), also ending at the value.
PARAPHRASE = {
    "city": ["If you visited {e} at home, you would be in", "The name of the city where {e} resides is"],
    "occupation": ["Asked what they do for a living, {e} would answer that they are a", "The job {e} holds is that of a"],
    "favorite object": ["The object {e} would save first from a fire is a", "Nothing {e} owns is loved more than their"],
    "sister": ["The name of {e}'s sister is", "If you asked {e} who their sister is, they would say"],
    "pet species": ["The animal {e} keeps as a pet is a", "Visitors to {e}'s home meet their pet, a"],
    "childhood fear": ["The thing that scared {e} most as a young child was", "In childhood, {e} feared"],
    "workplace": ["The place {e} goes to work is called", "If you wanted to find {e} at work, you would go to"],
    "hobby": ["What {e} likes to do in their spare time is", "The activity {e} most enjoys outside work is"],
    "birth month": ["The month of {e}'s birthday is", "{e} turns a year older every year in the month of"],
    "allergy": ["The substance {e} cannot be near because of an allergy is", "{e} has to avoid"],
}
COMPOSE = {  # "<attr> of {a}'s sister" two-hop, ends at the value
    a: t for a, t in {
        "city": "{a}'s sister lives in the city of", "occupation": "{a}'s sister works as a",
        "favorite object": "{a}'s sister's favorite object is a", "pet species": "{a}'s sister keeps a pet",
        "childhood fear": "As a child, {a}'s sister was afraid of", "workplace": "{a}'s sister works at",
        "hobby": "{a}'s sister's hobby is", "birth month": "{a}'s sister was born in the month of",
        "allergy": "{a}'s sister is allergic to", "sister": "The sister of {a}'s sister is",
    }.items()}
USE = {  # free-form "use" probe (reported only): a correct continuation should contain the value
    "city": "Write one sentence inviting {e} to a party in their own home city:",
    "occupation": "Write one sentence asking {e} for advice about their job:",
    "favorite object": "Write one sentence describing a gift for {e} that matches their favorite object:",
    "sister": "Write one sentence greeting {e}'s sister by name:",
    "pet species": "Write one sentence asking {e} how their pet is doing, naming the animal:",
    "childhood fear": "Write one sentence about what {e} was afraid of as a child:",
    "workplace": "Write one sentence asking {e} how things are at their workplace, naming it:",
    "hobby": "Write one sentence suggesting {e} spend Saturday on their hobby:",
    "birth month": "Write one sentence planning {e}'s birthday, naming the month:",
    "allergy": "Write one sentence warning a cook about {e}'s allergy:",
}

ONSETS = ["k", "v", "th", "z", "m", "qu", "dr", "l", "s", "r", "br", "n", "f", "gr", "y", "p", "sh", "t"]
NUCLEI = ["a", "e", "i", "o", "u", "ae", "ei", "y", "ou", "ia"]
CODAS = ["", "n", "r", "l", "s", "th", "x", "m", "v", "nd", "sk", "rn"]
SUFFIX = {  # gives each value pool a consistent "shape" without real-world meaning
    "city": ["mere", "holt", "vale", "spire", "wick"], "occupation": ["wright", "smith", "ward", "monger", "seer"],
    "favorite object": ["ite", "glass", "stone", "lyre", "orb"], "pet species": ["ling", "hound", "moth", "fin", "rook"],
    "childhood fear": ["shade", "maw", "drift", "hollow", "fen"], "workplace": [" Works", " Hall", " Yard", " Mill", " House"],
    "hobby": ["ing", "craft", "play", "weave", "song"], "birth month": ["ember", "ary", "uary", "ust", "il"],
    "allergy": ["dust", "pollen", "root", "seed", "bloom"],
}


def _syl(rng):
    return rng.choice(ONSETS) + rng.choice(NUCLEI) + rng.choice(CODAS)


def _word(rng, n):
    return "".join(_syl(rng) for _ in range(n))


@dataclass
class Fact:
    fid: str
    life: int
    entity: str
    attr: str
    value: str
    train_texts: list
    bridge: bool = False


@dataclass
class Stream:
    seed: int
    n_lives: int
    facts_per_life: int
    entities: list
    pools: dict
    lives: list = field(default_factory=list)          # list of lists of Fact
    composition: list = field(default_factory=list)    # {"prompt","answer","hops":[fid_a,fid_b],"life_b"}
    negatives: list = field(default_factory=list)      # {"prompt","sister_fid","attr"}

    def facts(self):
        return [f for life in self.lives for f in life]

    def to_json(self):
        d = asdict(self)
        return json.dumps(d, indent=1)


def make_stream(seed: int, n_lives: int = 100, facts_per_life: int = 10, n_entities: int = 300, n_new: int = 2,
                pool_per_attr: int = 50, bridges=(2, 3)) -> Stream:
    rng = random.Random(seed)
    names = set()
    while len(names) < n_entities:
        first, last = _word(rng, 2).capitalize(), _word(rng, rng.choice((1, 2))).capitalize()
        if first.lower() != "david" and len(first) >= 4 and len(last) >= 3:
            names.add(f"{first} {last}")
    entities = sorted(names)
    rng.shuffle(entities)
    pools = {}
    for a in ATTRS:
        if a in RELATIONAL:
            continue
        vals = set()
        while len(vals) < pool_per_attr:
            vals.add(_word(rng, rng.choice((1, 2))) + rng.choice(SUFFIX[a]))
        pools[a] = sorted(vals)
    pools["sister"] = entities

    st = Stream(seed, n_lives, facts_per_life, entities, pools)
    written = {}                 # (entity, attr) -> Fact
    introduced = {}              # entity -> life it first appeared (as a subject or as a sister value)
    queue = list(entities)       # entities never mentioned yet
    # 2 new subjects per life (200) plus ~55 sister values need > 200 names; 300 keeps the 2-3 bridge quota exact

    def free_attrs(e, used):
        return [a for a in ATTRS if (e, a) not in written and (e, a) not in used]

    def make_fact(life, idx, e, a, used, bridge):
        if a in RELATIONAL:
            if not queue:
                return None
            v = queue.pop(0)                       # the sister is a new person, mentioned here for the first time
            introduced.setdefault(v, life)
        else:
            taken = {f.value for (ee, _), f in written.items() if ee == e} | {v for (ee, aa), v in used.items() if ee == e}
            v = rng.choice([x for x in pools[a] if x not in taken])
        used[(e, a)] = v
        return Fact(f"L{life:03d}F{idx:02d}", life, e, a, v, [t.format(e=e, v=v) for t in TRAIN[a]], bridge=bridge)

    for life in range(1, n_lives + 1):
        facts, used = [], {}
        # 1) bridge facts: entities from EARLIER lives, preferring sisters that have no facts of their own yet
        n_bridge = rng.randint(*bridges) if life > 1 else 0
        old = [e for e, l in introduced.items() if l < life and free_attrs(e, used)]
        rng.shuffle(old)
        old.sort(key=lambda e: sum(1 for (ee, _) in written if ee == e))   # least-described first
        for e in old[:n_bridge]:
            a = rng.choice([x for x in free_attrs(e, used) if x not in RELATIONAL] or free_attrs(e, used))
            f = make_fact(life, len(facts), e, a, used, bridge=True)
            if f:
                facts.append(f)
        # 2) this life's own people: new subjects (plus David while he has unwritten attributes)
        subjects = [queue.pop(0) for _ in range(min(n_new, len(queue)))]
        for e in subjects:
            introduced.setdefault(e, life)
        if free_attrs(SUBJECT, used) and (life == 1 or rng.random() < 0.15):
            subjects.append(SUBJECT)
            introduced.setdefault(SUBJECT, life)
        tries = 0
        while len(facts) < facts_per_life and tries < 2000:
            tries += 1
            cands = [e for e in subjects if free_attrs(e, used)]
            if not cands:   # everyone in this life is fully described: fall back to any earlier entity (not a bridge quota)
                cands = [e for e, l in introduced.items() if free_attrs(e, used)]
                if not cands:
                    break
            e = rng.choice(cands)
            f = make_fact(life, len(facts), e, rng.choice(free_attrs(e, used)), used,
                          bridge=introduced.get(e, life) < life)
            if f:
                facts.append(f)
        for f in facts:
            written[(f.entity, f.attr)] = f
        st.lives.append(facts)

    # composition: A sister B (life i) + B attr V (life j > i); negative: A sister B with B attr never written
    for (a_ent, attr), sf in written.items():
        if attr != "sister":
            continue
        b = sf.value
        for battr in ATTRS:
            bf = written.get((b, battr))
            if bf is not None and bf.life > sf.life:
                st.composition.append({"prompt": COMPOSE[battr].format(a=a_ent), "answer": bf.value,
                                       "hops": [sf.fid, bf.fid], "life_b": bf.life})
            elif bf is None and battr != "sister":
                st.negatives.append({"prompt": COMPOSE[battr].format(a=a_ent), "sister_fid": sf.fid, "attr": battr})
    rng.shuffle(st.negatives)
    st.negatives = st.negatives[: max(len(st.composition), 1)]
    return st


def probes_for(f: Fact) -> dict:
    """Template (cut training phrasing 0), held-out paraphrases, and the reported-only use probe."""
    t0 = TRAIN[f.attr][0]
    return {"template": t0[: t0.index("{v}")].format(e=f.entity).rstrip(),
            "paraphrase": [p.format(e=f.entity) for p in PARAPHRASE[f.attr]],
            "use": USE[f.attr].format(e=f.entity), "answer": f.value}
