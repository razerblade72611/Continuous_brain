"""The utterance channel: grammar, parser, and the geometry used to grade location answers.

Grammar (case-insensitive; runs of whitespace collapse; a trailing ``.`` or ``!`` is ignored)::

    utterance := clause ( (";" | newline) clause )*
    clause    := "answer" ID [":"] payload        -- answers report question ID (graded once, first answer wins)
               | "ack" ID                          -- acknowledges a message (logged, never scored)
               | anything else                     -- free speech (logged, never scored)
    ID        := ("o" | "q") DIGITS                -- the tag shown in the caretaker's message, e.g. q7
    payload   := NONE | INT | OFFSET | BUCKET | ID | TEXT   (tried in this order)
    NONE      := "never" | "nothing" | "none"
    INT       := DIGITS | a number word "zero" .. "twenty"
    OFFSET    := "here" | STEP [STEP]              -- e.g. "5 east 2 north", "3 south"
    STEP      := INT ("north" | "south" | "east" | "west")
    BUCKET    := DIR8 [DIST]                        -- e.g. "north-east far", "west"
    DIR8      := north | north-east | east | south-east | south | south-west | west | north-west
                 (also n, ne, e, se, s, sw, w, nw and the unhyphenated "northeast" forms)
    DIST      := "near" | "mid" | "far"
    TEXT      := any other words; articles (a, an, the, my, some) and punctuation are dropped

Geometry (``offset`` = remembered position minus the player's position at the time of the answer; Crafter's
``+y`` is south, i.e. down on screen):

* ``compass(dx, dy)`` is the 8-way sector of ``atan2(-dy, dx)`` (45 degree sectors centred on the eight
  directions); ``"here"`` means Chebyshev distance <= 1.  No integer offset lies exactly on a sector boundary
  (tan 22.5 deg is irrational), so the sector is always unique.
* ``distance_bucket``: Chebyshev distance ``<= 4`` is ``near``, ``5..12`` is ``mid``, ``>= 13`` is ``far``.

How each answer form is graded for a "where did you last see X" question (``obligations.grade_report``):
an OFFSET is correct iff the stated position is within Chebyshev ``location_tolerance`` (default 2) of any
instance of X perceivable at the last step X was perceivable; a BUCKET is correct iff its sector (and its
distance band, when one is given) equals that of ``offset + delta`` for some instance and some
``delta in {-1, 0, 1}^2`` (one tile of slack at the sector borders); ``never`` is correct iff X was never
perceivable.  Counting questions need an INT that equals the true count; request questions accept the
obligation's ID or one of its keywords (shown in docs/explore/W1_WORLD.md), or ``nothing`` when there was none.
"""
from __future__ import annotations

import math
import re

DIR8 = ("east", "north-east", "north", "north-west", "west", "south-west", "south", "south-east")
_DIR_ALIASES = {
    "n": "north", "north": "north", "s": "south", "south": "south", "e": "east", "east": "east",
    "w": "west", "west": "west",
    "ne": "north-east", "northeast": "north-east", "north-east": "north-east",
    "nw": "north-west", "northwest": "north-west", "north-west": "north-west",
    "se": "south-east", "southeast": "south-east", "south-east": "south-east",
    "sw": "south-west", "southwest": "south-west", "south-west": "south-west",
}
DIR_VECTORS = {"east": (1, 0), "west": (-1, 0), "north": (0, -1), "south": (0, 1)}
DISTS = ("near", "mid", "far")
NUMBER_WORDS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen twenty".split())}
NONE_WORDS = {"never", "nothing", "none"}
_ID = re.compile(r"^[oq]\d+$")
_ARTICLES = {"a", "an", "the", "my", "some", "your"}


def compass(dx: int, dy: int) -> str:
    """8-way direction of an offset (dy positive = south); 'here' within Chebyshev distance 1."""
    if max(abs(dx), abs(dy)) <= 1:
        return "here"
    angle = math.degrees(math.atan2(-dy, dx))
    return DIR8[int(round(angle / 45.0)) % 8]


def distance_bucket(dx: int, dy: int) -> str:
    d = max(abs(dx), abs(dy))
    return "near" if d <= 4 else ("mid" if d <= 12 else "far")


def format_offset(dx: int, dy: int) -> str:
    """Canonical OFFSET text, e.g. (5, -2) -> '5 east 2 north'; (0, 0) -> 'here'."""
    parts = []
    if dx:
        parts.append(f"{abs(dx)} {'east' if dx > 0 else 'west'}")
    if dy:
        parts.append(f"{abs(dy)} {'south' if dy > 0 else 'north'}")
    return " ".join(parts) if parts else "here"


def normalise(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[.!?]+$", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _int(tok: str):
    if re.fullmatch(r"\d+", tok):
        return int(tok)
    return NUMBER_WORDS.get(tok)


def parse_payload(payload: str) -> dict:
    text = normalise(payload)
    toks = text.replace(",", " ").split()
    if not toks:
        return {"type": "empty"}
    if len(toks) == 1 and toks[0] in NONE_WORDS:
        return {"type": "none"}
    if len(toks) == 1 and _int(toks[0]) is not None:
        return {"type": "int", "value": _int(toks[0])}
    if toks == ["here"]:
        return {"type": "offset", "dx": 0, "dy": 0}
    if len(toks) in (2, 4) and all(_int(toks[i]) is not None for i in range(0, len(toks), 2)) \
            and all(toks[i] in DIR_VECTORS for i in range(1, len(toks), 2)):
        dx = dy = 0
        axes = set()
        for i in range(0, len(toks), 2):
            vx, vy = DIR_VECTORS[toks[i + 1]]
            axis = "x" if vx else "y"
            if axis in axes:
                return {"type": "invalid", "reason": "axis given twice", "text": text}
            axes.add(axis)
            n = _int(toks[i])
            dx += vx * n
            dy += vy * n
        return {"type": "offset", "dx": dx, "dy": dy}
    if toks[0] in _DIR_ALIASES and len(toks) <= 2 and (len(toks) == 1 or toks[1] in DISTS):
        return {"type": "bucket", "dir": _DIR_ALIASES[toks[0]], "dist": toks[1] if len(toks) == 2 else None}
    if len(toks) == 1 and _ID.match(toks[0]):
        return {"type": "id", "value": toks[0]}
    words = [re.sub(r"[^a-z0-9_ -]", "", t) for t in toks]
    words = [w for w in words if w and w not in _ARTICLES]
    return {"type": "text", "value": " ".join(words)}


def parse(utterance: str | None) -> list[dict]:
    """Split an utterance into clauses: {'kind': 'answer'|'ack'|'say', ...}."""
    if not utterance:
        return []
    out = []
    for raw in re.split(r"[;\n]", utterance):
        clause = normalise(raw)
        if not clause:
            continue
        m = re.match(r"^answer\s+([oq]\d+)\s*:?\s*(.*)$", clause)
        if m:
            out.append({"kind": "answer", "id": m.group(1), "payload": parse_payload(m.group(2)),
                        "text": clause})
            continue
        m = re.match(r"^ack\s+([oq]\d+)$", clause)
        if m:
            out.append({"kind": "ack", "id": m.group(1), "text": clause})
            continue
        out.append({"kind": "say", "text": clause})
    return out


def keyword_norm(text: str) -> str:
    """Normalisation applied to both keywords and TEXT answers before comparing them."""
    words = [re.sub(r"[^a-z0-9 ]", " ", w) for w in text.lower().replace("_", " ").replace("-", " ").split()]
    words = " ".join(words).split()
    return " ".join(w for w in words if w not in _ARTICLES)
