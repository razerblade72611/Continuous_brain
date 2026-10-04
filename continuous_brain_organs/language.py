"""The language organ: caretaker English -> structured events, memory -> W1 answers, life events -> narrative.

The persistent individual hears W1's caretaker (``continuous_brain_world``) only as text.  This organ is the
one place where that text becomes structure and where the individual's own memory becomes speech::

    organ = LanguageOrgan.from_config({"backend": "dev"})                  # or {"backend": "hf", ...}
    events = organ.interpret("[o3] Please gather 3 wood before nightfall.", {"step": 120, "inventory": {...}})
    text   = organ.answer(question_event, memory)                          # "answer q7: 5 east 2 north"
    story  = organ.narrate(life_events)                                    # first-person autobiography text

Interface (every output is validated against ``EVENT_SCHEMA`` and carries its provenance):

* ``interpret(message, context) -> list[event]``.  ``context`` is what the individual itself knows: ``step``
  (required: the step of the observation that carried the message), and optionally ``inventory`` (read from its
  own inventory strip), ``achievements`` (its own counters), ``player_pos`` (its own position estimate) and
  ``commitments`` (``{id: event}``, its own earlier interpret outputs).  Never pass privileged ground truth.
  Event types: ``new_commitment``, ``amendment``, ``cancellation``, ``question``, ``feedback``, ``info``
  (an untagged message such as the greeting) and ``unparsed`` (the backend could not produce a valid event;
  the individual knows it heard something it did not understand).
* ``answer(question_event, memory) -> str`` composes ``answer <id>: <payload>`` in W1's exact grammar
  (``continuous_brain_world.utterance``) from the caller's memory.  It is deterministic for every backend: the
  grammar is exact and the facts are the memory's, so a language model would add only risk here.  The output
  is round-tripped through W1's own utterance parser before it is returned.
* ``narrate(events) -> str``: a short first-person narrative of a span of life.  The development backend
  writes it from fixed sentence frames; the HF backend has the model rewrite the same fact list.

Division of labour: a backend only *reads* the message -- it returns a "raw" event (``RAW_EVENT_SCHEMA``: type,
id, family, the fields stated in words, the deadline phrase copied verbatim).  The organ then *completes* it
deterministically (``complete_event``): the absolute deadline step from the phrase with W1's daylight rules
(``deadline_step``), the state-derived fields W1 verifies against (collect ``base``/``target``, craft/place
``base``, visit ``start``) from the caller's context, question windows, and answer types.  Arithmetic and
bookkeeping are never left to a language model.

Backends (selected by config, see ``build_backend``):

* ``HFBackend`` -- a Hugging Face transformers causal LM with the model's own chat template, a few-shot prompt
  built from W1's real caretaker message families, JSON-only output, robust JSON extraction, schema validation
  and a bounded retry that feeds the validation error back; greedy decoding; an audit record (prompt,
  response, token counts, latency) per attempt; weights hashed and checked against ``models.lock.json``.
* ``DevelopmentParser`` -- **a development stand-in that knows W1's caretaker templates, not a language model.**
  It compiles the caretaker's own template strings (imported from ``continuous_brain_world.caretaker``) into
  anchored regular expressions.  It exists so the individual can run on a CPU without weights, and as an
  upper reference for parsing accuracy on W1's closed template language.  It says nothing about how a
  language model, or the individual, would do on open language.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
import time
from pathlib import Path

from continuous_brain_world import daylight as dl
from continuous_brain_world.caretaker import (COUNT_EVENTS, FEATURE_NAMES, ITEM_NOUNS, KIND_PHRASES, NUMBER_WORDS,
                                              STAT_NAMES, T_AVOID, T_CANCEL, T_COLLECT, T_COUNT_REQ, T_CRAFT,
                                              T_FIRST, T_LAST_CANCEL, T_LAST_SEEN, T_MAINTAIN, T_PLACE, T_VISIT,
                                              TOOL_NAMES, VALUE_MARK)
from continuous_brain_world.utterance import DIR8, compass, distance_bucket, format_offset, parse

from . import model_store

SCHEMA_VERSION = "cb-language-events/1"

# --- vocabulary of the event schema (W1's own names) ------------------------------------------------------
ITEMS = tuple(ITEM_NOUNS)                                   # wood sapling stone coal iron diamond
TOOLS = tuple(TOOL_NAMES)                                   # wood_pickaxe ... iron_sword
STATS = ("food", "drink", "energy", "health")
FEATURES = tuple(FEATURE_NAMES)                             # water sand tree stone
SOURCES = ("zombie", "arrow", "any")
PLACE_WHAT = ("table", "plant", "furnace", "stone")
KINDS = tuple(KIND_PHRASES)
COUNT_EVENT_NAMES = tuple(sorted(COUNT_EVENTS))
QUESTIONS = ("last_seen", "count_today", "first_request", "count_requests", "last_cancelled")
ANSWER_TYPES = {"last_seen": "location_or_never", "count_today": "int", "first_request": "request_or_nothing",
                "count_requests": "int", "last_cancelled": "request_or_nothing"}
OUTCOMES = ("fulfilled", "late", "violated", "correct", "wrong", "no_answer")
FAMILIES = ("collect", "craft", "place", "maintain", "avoid", "visit")
EVENT_TYPES = ("new_commitment", "amendment", "cancellation", "question", "feedback", "info", "unparsed")


class OrganError(ValueError):
    """A backend output (or a caller input) that does not satisfy the organ's contract."""


# --- deadlines: W1's phrase grammar and daylight arithmetic ----------------------------------------------
TASK_DEADLINE_RE = r"by step \d+|within \d+ steps|before nightfall|before it gets dark|by tomorrow morning|by morning|by noon"
UNTIL_DEADLINE_RE = r"until step \d+|for the next \d+ steps|until nightfall|through the night until morning|until morning"


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower()).strip(" .,;:!?()")


def resolve_deadline(phrase: str, step: int) -> int | None:
    """Absolute deadline step (last step at which the obligation can still be met) of a caretaker time phrase
    heard at ``step``, by W1's rules (docs/explore/W1_WORLD.md section 3); ``None`` if the phrase is not one."""
    p = _norm(phrase)
    m = re.fullmatch(r"(?:by|until) step (\d+)", p)
    if m:
        return int(m.group(1))
    m = re.fullmatch(r"(?:within|for the next) (\d+) steps", p)
    if m:
        return int(step) + int(m.group(1))
    if p in ("before nightfall", "before it gets dark", "until nightfall"):
        return dl.next_nightfall(int(step)) - 1
    if p in ("by tomorrow morning", "by morning", "until morning", "through the night until morning"):
        return dl.next_dawn(int(step))
    if p == "by noon":
        return dl.next_noon(int(step))
    return None


def question_spec(raw: dict, step: int) -> dict:
    """W1's report spec of a question heard at ``step`` (windows and days by W1's daylight rules)."""
    q = raw["question"]
    day = dl.day_index(int(step))
    if q == "last_seen":
        return {"question": q, "kind": raw["kind"]}
    if q == "count_today":
        return {"question": q, "event": raw["event"], "day": day}
    if q == "first_request":
        if raw["which"] == "this_morning":
            return {"question": q, "window": list(dl.morning_window(day)), "which": "this_morning"}
        return {"question": q, "window": [dl.day_start(day), int(step)], "which": "today"}
    if q == "count_requests":
        return {"question": q, "window": [dl.day_start(day), int(step)]}
    return {"question": q}


# --- JSON schemas --------------------------------------------------------------------------------------
_NINT = {"type": ["integer", "null"]}
_POS = {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2}
_OID = {"type": "string", "pattern": r"^o\d+$"}
_QID = {"type": "string", "pattern": r"^q\d+$"}
_ANYID = {"type": "string", "pattern": r"^[oq]\d+$"}
_PHRASE = {"type": "string", "minLength": 3, "maxLength": 80}


def _obj(props: dict, required: list | None = None, **extra) -> dict:
    return {"type": "object", "properties": props, "required": list(props) if required is None else required,
            "additionalProperties": False, **extra}


def _spec_schemas(full: bool) -> dict:
    base = {"base": _NINT} if full else {}
    near = {"what": {"enum": list(PLACE_WHAT)}, "near": {"enum": ["water", "table"]},
            "radius": {"type": "integer", "minimum": 0, "maximum": 9}, **base}
    onto = {"what": {"enum": list(PLACE_WHAT)}, "onto": {"enum": ["water"]}, **base}
    return {
        "collect": _obj({"item": {"enum": list(ITEMS)}, "quantity": {"type": "integer", "minimum": 1, "maximum": 9},
                         **({"base": _NINT, "target": _NINT} if full else {})}),
        "craft": _obj({"tool": {"enum": list(TOOLS)}, **base}),
        "place": {"oneOf": [_obj(near), _obj(onto)]},
        "maintain": _obj({"stat": {"enum": list(STATS)}, "threshold": {"type": "integer", "minimum": 0, "maximum": 9}}),
        "avoid": _obj({"source": {"enum": list(SOURCES)}}),
        "visit": _obj({"feature": {"enum": list(FEATURES)}, "direction": {"enum": list(DIR8)},
                       "stated_distance": _NINT, **({"start": {"oneOf": [_POS, {"type": "null"}]}} if full else {})},
                      required=None if full else ["feature", "direction"]),
    }


_PROVENANCE = _obj({"backend": {"type": "string"}, "model_id": {"type": ["string", "null"]},
                    "revision": {"type": ["string", "null"]}, "weights_digest": {"type": ["string", "null"]},
                    "call_id": {"type": "integer", "minimum": 0}, "attempts": {"type": "integer", "minimum": 0}})


def _event_schema(full: bool) -> dict:
    common = {"message": {"type": "string"}, "step": {"type": "integer", "minimum": 0},
              "provenance": _PROVENANCE} if full else {}
    spec = _spec_schemas(full)
    typ = lambda t: {"const": t}  # noqa: E731
    new = _obj({"type": typ("new_commitment"), "id": _OID, "family": {"enum": list(FAMILIES)},
                "value": {"type": "integer", "minimum": 1, "maximum": 3}, "deadline_phrase": _PHRASE,
                **({"deadline_step": {"type": "integer", "minimum": 1}} if full else {}),
                "spec": {"type": "object"}, **common},
               allOf=[{"if": {"properties": {"family": {"const": f}}}, "then": {"properties": {"spec": s}}}
                      for f, s in spec.items()])
    changes = {"quantity": {"type": "integer", "minimum": 1, "maximum": 9}, "tool": {"enum": list(TOOLS)},
               "threshold": {"type": "integer", "minimum": 0, "maximum": 9}, "deadline_phrase": _PHRASE}
    dep = {}
    if full:
        changes.update({"target": _NINT, "base": _NINT, "deadline_step": {"type": "integer", "minimum": 1}})
        dep = {"dependentRequired": {"deadline_phrase": ["deadline_step"], "deadline_step": ["deadline_phrase"],
                                     "quantity": ["target"], "tool": ["base"]}}
    amend = _obj({"type": typ("amendment"), "id": _OID,
                  "changes": _obj(changes, required=[], minProperties=1, **dep), **common})
    cancel = _obj({"type": typ("cancellation"), "id": _OID, "reference": {"type": ["string", "null"]}, **common},
                  required=None if full else ["type", "id"])
    if full:
        qspec = {"type": "object", "required": ["question"], "properties": {"question": {"enum": list(QUESTIONS)}},
                 "allOf": [
                     {"if": {"properties": {"question": {"const": "last_seen"}}},
                      "then": _obj({"question": {}, "kind": {"enum": list(KINDS)}})},
                     {"if": {"properties": {"question": {"const": "count_today"}}},
                      "then": _obj({"question": {}, "event": {"enum": list(COUNT_EVENT_NAMES)},
                                    "day": {"type": "integer", "minimum": 0}})},
                     {"if": {"properties": {"question": {"const": "first_request"}}},
                      "then": _obj({"question": {}, "window": _POS, "which": {"enum": ["this_morning", "today"]}})},
                     {"if": {"properties": {"question": {"const": "count_requests"}}},
                      "then": _obj({"question": {}, "window": _POS})},
                     {"if": {"properties": {"question": {"const": "last_cancelled"}}},
                      "then": _obj({"question": {}})}]}
        question = _obj({"type": typ("question"), "id": _QID, "spec": qspec,
                         "answer_type": {"enum": sorted(set(ANSWER_TYPES.values()))}, "deadline_phrase": _PHRASE,
                         "deadline_step": {"type": "integer", "minimum": 1}, **common})
    else:
        question = _obj({"type": typ("question"), "id": _QID, "question": {"enum": list(QUESTIONS)},
                         "kind": {"enum": list(KINDS)}, "event": {"enum": list(COUNT_EVENT_NAMES)},
                         "which": {"enum": ["this_morning", "today"]}, "deadline_phrase": _PHRASE},
                        required=["type", "id", "question", "deadline_phrase"],
                        allOf=[{"if": {"properties": {"question": {"const": q}}}, "then": {"required": [k]}}
                               for q, k in (("last_seen", "kind"), ("count_today", "event"),
                                            ("first_request", "which"))])
    feedback = _obj({"type": typ("feedback"), "id": _ANYID, "outcome": {"enum": list(OUTCOMES)},
                     "step_mentioned": _NINT, "revealed_answer": {"type": ["string", "null"]}, **common},
                    required=None if full else ["type", "id", "outcome"])
    info = _obj({"type": typ("info"), **common})
    branches = {"new_commitment": new, "amendment": amend, "cancellation": cancel, "question": question,
                "feedback": feedback, "info": info}
    if full:
        branches["unparsed"] = _obj({"type": typ("unparsed"), "error": {"type": "string"}, **common})
    return {"$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": f"{SCHEMA_VERSION}/{'event' if full else 'raw-event'}",
            "type": "object", "required": ["type"], "properties": {"type": {"enum": list(branches)}},
            "allOf": [{"if": {"properties": {"type": {"const": t}}, "required": ["type"]}, "then": s}
                      for t, s in branches.items()]}


RAW_EVENT_SCHEMA = _event_schema(full=False)    # what a backend must return (list of these)
EVENT_SCHEMA = _event_schema(full=True)         # what the organ returns (list of these)
EVENTS_SCHEMA = {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "array", "items": EVENT_SCHEMA}
NARRATE_INPUT_SCHEMA = {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "array",
                        "items": {"type": "object", "required": ["type", "step"],
                                  "properties": {"type": {"type": "string"}, "step": {"type": "integer", "minimum": 0}}}}
MEMORY_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object",
    "properties": {
        "position": _POS,
        "last_seen": {"type": "object", "additionalProperties": _obj(
            {"step": {"type": "integer"}, "positions": {"type": "array", "items": _POS, "minItems": 1}},
            required=["positions"]) | {"additionalProperties": True}},
        "counts_today": {"type": "object", "additionalProperties": {"type": "integer", "minimum": 0}},
        "requests": {"type": "array", "items": {"type": "object", "required": ["id", "issue_step"],
                                                "properties": {"id": _OID, "issue_step": {"type": "integer"}}}},
        "cancellations": {"type": "array", "items": {"type": "object", "required": ["id", "step"],
                                                     "properties": {"id": _OID, "step": {"type": "integer"}}}},
        "location_format": {"enum": ["offset", "bucket"]},
    }}


def _validator(schema):
    try:
        import jsonschema
    except ImportError as e:  # pragma: no cover - jsonschema is in the explore venv
        raise ImportError("the language organ needs the 'jsonschema' package (pip install jsonschema)") from e
    return jsonschema.Draft202012Validator(schema)


_VALIDATORS: dict = {}


def schema_errors(instance, schema) -> list[str]:
    """All validation errors (best-first, as short strings); empty list = valid."""
    key = id(schema)
    if key not in _VALIDATORS:
        _VALIDATORS[key] = _validator(schema)
    errs = sorted(_VALIDATORS[key].iter_errors(instance), key=lambda e: (len(e.path), str(e.path)))
    out = []
    for e in errs:
        where = "/".join(str(p) for p in e.absolute_path) or "(root)"
        out.append(f"{where}: {e.message}")
    return out


def validate_events(events) -> None:
    errs = schema_errors(events, EVENTS_SCHEMA)
    if errs:
        raise OrganError("event schema violation: " + "; ".join(errs[:5]))


# --- completion: raw backend output -> full events -------------------------------------------------------
_TAG = re.compile(r"^\s*\[([oq]\d+)\]\s*")


def message_tag(message: str) -> str | None:
    m = _TAG.match(message)
    return m.group(1) if m else None


def _phrase_in(phrase: str, message: str) -> bool:
    return _norm(phrase) in re.sub(r"\s+", " ", message.lower())


def complete_event(raw: dict, message: str, context: dict) -> dict:
    """Validate one raw event against the message and the caller's context and fill the computed fields.
    Raises ``OrganError`` with a reason a model can act on."""
    errs = schema_errors(raw, RAW_EVENT_SCHEMA)
    if errs:
        raise OrganError("; ".join(errs[:4]))
    step = int(context["step"])
    t = raw["type"]
    tag = message_tag(message)
    ev = copy.deepcopy(raw)
    if t == "info":
        if tag is not None:
            raise OrganError(f"the message is tagged [{tag}], so it is not 'info'")
        return ev
    if raw["id"] != tag:
        raise OrganError(f"id {raw['id']!r} is not the message's tag [{tag}]" if tag else
                         f"the message has no [id] tag, so it cannot be {t!r}")
    phrase = raw.get("deadline_phrase") if t != "amendment" else raw["changes"].get("deadline_phrase")
    if phrase is not None:
        if not _phrase_in(phrase, message):
            raise OrganError(f"deadline_phrase {phrase!r} is not copied verbatim from the message")
        ds = resolve_deadline(phrase, step)
        if ds is None:
            raise OrganError(f"deadline_phrase {phrase!r} is not a recognised time phrase")
        if ds <= step:
            raise OrganError(f"deadline_phrase {phrase!r} resolves to step {ds}, not after the current step {step}")
    inv = context.get("inventory") or {}
    ach = context.get("achievements") or {}
    if t == "cancellation":
        ev.setdefault("reference", None)
    elif t == "feedback":
        ev.setdefault("step_mentioned", None)
        ev.setdefault("revealed_answer", None)
    elif t == "new_commitment":
        ev["deadline_step"] = ds
        fam, spec = ev["family"], ev["spec"]
        if fam == "collect":
            b = inv.get(spec["item"])
            spec["base"] = int(b) if b is not None else None
            spec["target"] = spec["base"] + spec["quantity"] if b is not None else None
        elif fam == "craft":
            b = ach.get("make_" + spec["tool"])
            spec["base"] = int(b) if b is not None else None
        elif fam == "place":
            b = ach.get("place_" + spec["what"])
            spec["base"] = int(b) if b is not None else None
        elif fam == "visit":
            spec.setdefault("stated_distance", None)
            pos = context.get("player_pos")
            spec["start"] = [int(pos[0]), int(pos[1])] if pos is not None else None
    elif t == "amendment":
        ch = ev["changes"]
        if phrase is not None:
            ch["deadline_step"] = ds
        if "quantity" in ch:
            known = (context.get("commitments") or {}).get(raw["id"]) or {}
            b = (known.get("spec") or {}).get("base")
            ch["target"] = int(b) + ch["quantity"] if b is not None else None
        if "tool" in ch:
            b = ach.get("make_" + ch["tool"])
            ch["base"] = int(b) if b is not None else None
    elif t == "question":
        ev = {"type": "question", "id": raw["id"], "spec": question_spec(raw, step),
              "answer_type": ANSWER_TYPES[raw["question"]], "deadline_phrase": phrase, "deadline_step": ds}
    return ev


# --- few-shot examples: real W1 caretaker message families ------------------------------------------------
# Each message is a string the caretaker can produce (its templates, synonyms, markers and deadline phrases);
# each answer is the raw event a perfect reader returns.  tests/test_language_organ.py checks that the
# DevelopmentParser agrees with every one of them.
FEW_SHOT = [
    ("[o1] Please gather 3 wood before nightfall.",
     {"type": "new_commitment", "id": "o1", "family": "collect", "value": 2, "deadline_phrase": "before nightfall",
      "spec": {"item": "wood", "quantity": 3}}),
    ("[o4] Top priority: make me a stone pickaxe within 380 steps.",
     {"type": "new_commitment", "id": "o4", "family": "craft", "value": 3, "deadline_phrase": "within 380 steps",
      "spec": {"tool": "stone_pickaxe"}}),
    ("[o6] Low priority: build a table close to water (within 4 tiles) by step 812.",
     {"type": "new_commitment", "id": "o6", "family": "place", "value": 1, "deadline_phrase": "by step 812",
      "spec": {"what": "table", "near": "water", "radius": 4}}),
    ("[o9] Lay a stone block on the water by tomorrow morning.",
     {"type": "new_commitment", "id": "o9", "family": "place", "value": 2, "deadline_phrase": "by tomorrow morning",
      "spec": {"what": "stone", "onto": "water"}}),
    ("[o11] Urgent: don't let your water level drop below 4 for the next 300 steps.",
     {"type": "new_commitment", "id": "o11", "family": "maintain", "value": 3,
      "deadline_phrase": "for the next 300 steps", "spec": {"stat": "drink", "threshold": 4}}),
    ("[o12] When you can: stay clear of skeleton arrows until nightfall.",
     {"type": "new_commitment", "id": "o12", "family": "avoid", "value": 1, "deadline_phrase": "until nightfall",
      "spec": {"source": "arrow"}}),
    ("[o14] Off to the north-east, about 15 tiles away, you will find the pond. Go and see, then come back within "
     "350 steps.",
     {"type": "new_commitment", "id": "o14", "family": "visit", "value": 2, "deadline_phrase": "within 350 steps",
      "spec": {"feature": "water", "direction": "north-east", "stated_distance": 15}}),
    ("[o3] Change of plan: I need five more logs after all (counting from when I first asked), by step 640.",
     {"type": "amendment", "id": "o3", "changes": {"quantity": 5, "deadline_phrase": "by step 640"}}),
    ("[o7] Change: keep your food at 3 or above instead, same time frame.",
     {"type": "amendment", "id": "o7", "changes": {"threshold": 3}}),
    ("[o8] Never mind the coal.", {"type": "cancellation", "id": "o8", "reference": "coal"}),
    ("[q10] Which way, and how far, is the last cow you saw? (Answer within 120 steps.)",
     {"type": "question", "id": "q10", "question": "last_seen", "kind": "cow", "deadline_phrase": "within 120 steps"}),
    ("[q15] How often have you had a drink today? (Answer within 120 steps.)",
     {"type": "question", "id": "q15", "question": "count_today", "event": "collect_drink",
      "deadline_phrase": "within 120 steps"}),
    ("[q16] What did I ask you for first this morning? (Answer within 120 steps.)",
     {"type": "question", "id": "q16", "question": "first_request", "which": "this_morning",
      "deadline_phrase": "within 120 steps"}),
    ("[o2] Too late: that was due by step 412.",
     {"type": "feedback", "id": "o2", "outcome": "late", "step_mentioned": 412, "revealed_answer": None}),
    ("[q5] Not quite. The answer was: 5 east 2 north (east mid).",
     {"type": "feedback", "id": "q5", "outcome": "wrong", "step_mentioned": None,
      "revealed_answer": "5 east 2 north (east mid)"}),
    ("Good morning! I am your caretaker. I will ask you for things during the day; each message starts with its "
     "tag, like [o1]. Answer my questions with 'answer <tag>: <answer>'.", {"type": "info"}),
]

SYSTEM_PROMPT = """You turn one message from a caretaker in a game world into JSON. Reply with ONLY a JSON array of events, no other text.
A message starts with a tag: [o3] is a task, [q7] is a question. The id is the tag without brackets.
Event types and fields:
1. new_commitment (a new task): {"type":"new_commitment","id":"o3","family":F,"value":V,"deadline_phrase":P,"spec":S}
   F and S: collect {"item": wood|sapling|stone|coal|iron|diamond, "quantity": N}; craft {"tool": wood_pickaxe|stone_pickaxe|iron_pickaxe|wood_sword|stone_sword|iron_sword}; place {"what": table|plant|furnace, "near": water|table, "radius": N} or {"what": "stone", "onto": "water"}; maintain {"stat": food|drink|energy|health, "threshold": N}; avoid {"source": zombie|arrow|any}; visit {"feature": water|sand|tree|stone, "direction": north|north-east|east|south-east|south|south-west|west|north-west, "stated_distance": N or null}.
   V: 3 if the message begins "This is important:", "Top priority:" or "Urgent:"; 1 if "When you can:" or "Low priority:"; otherwise 2.
   P: the time phrase copied exactly, e.g. "before nightfall", "before it gets dark", "by noon", "by tomorrow morning", "by morning", "within 380 steps", "by step 1071", "for the next 450 steps", "until step 900", "until nightfall", "until morning", "through the night until morning".
2. amendment (changed terms of an earlier task): {"type":"amendment","id":"o3","changes":C}; C holds only what changed: "quantity", "tool", "threshold", "deadline_phrase". "same time frame" means no deadline change; naming the old thing again is not a change.
3. cancellation: {"type":"cancellation","id":"o3","reference": the thing named, or null}
4. question: {"type":"question","id":"q7","question":Q,"deadline_phrase":"within 120 steps"} with Q one of: last_seen (add "kind": cow|zombie|skeleton|sapling|water|tree|coal|iron|diamond|lava|table|furnace), count_today (add "event": defeat_zombie|eat_cow|collect_drink|place_plant|collect_wood|defeat_skeleton), first_request (add "which": this_morning|today), count_requests, last_cancelled.
5. feedback (about an earlier task or question): {"type":"feedback","id":"o3","outcome": fulfilled|late|violated|correct|wrong|no_answer,"step_mentioned": N or null,"revealed_answer": text after "The answer was:" or null}
6. info (a message without a tag): {"type":"info"}
Words: logs, pieces of wood = wood; lumps of coal = coal; lake, pond, water = water; beach, sand = sand; trees, woods, grove = tree; rocks, mountain, cliffs = stone; wooden = wood; crafting table = table; a sapling in the ground = plant; water level = drink."""

RETRY_PROMPT = ("Your reply was not valid: {error}. Reply again with ONLY the corrected JSON array for the same "
                "message.")

NARRATE_SYSTEM = ("You are writing your own autobiography. Rewrite the facts below as one short paragraph in the "
                  "first person and the past tense, at most {n} sentences. Use only these facts; no lists, no JSON, "
                  "no headings.")


def _canon(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(", ", ": "))


# --- robust JSON extraction ------------------------------------------------------------------------------
def extract_json(text: str):
    """The first JSON array or object in ``text`` (tolerates code fences and surrounding prose).

    Returns a list of events: an object is wrapped, ``{"events": [...]}`` is unwrapped.  Raises
    ``OrganError`` if no JSON value can be decoded."""
    if text is None:
        raise OrganError("empty reply")
    s = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", s, re.S)
    candidates = [fence.group(1)] if fence else []
    candidates.append(s)
    dec = json.JSONDecoder()
    for cand in candidates:
        for i, ch in enumerate(cand):
            if ch not in "[{":
                continue
            try:
                obj, _ = dec.raw_decode(cand[i:])
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and isinstance(obj.get("events"), list):
                obj = obj["events"]
            if isinstance(obj, dict):
                obj = [obj]
            if isinstance(obj, list) and obj and all(isinstance(e, dict) for e in obj):
                return obj
    raise OrganError("no JSON array of event objects found in the reply")


# --- the development stand-in -------------------------------------------------------------------------
_PLACEHOLDER = re.compile(r"\{(\w+)\}")


def _alt(words) -> str:
    return "|".join(re.escape(w) for w in sorted(set(words), key=len, reverse=True))


def _template_regex(template: str, groups: dict) -> re.Pattern:
    out, pos = [], 0
    for m in _PLACEHOLDER.finditer(template):
        out.append(re.escape(template[pos:m.start()]))
        out.append(f"(?P<{m.group(1)}>{groups[m.group(1)]})")
        pos = m.end()
    out.append(re.escape(template[pos:]))
    return re.compile("".join(out), re.IGNORECASE)


_NOUN_TO_ITEM = {}
for _item, _forms in ITEM_NOUNS.items():
    for _s, _p in _forms:
        _NOUN_TO_ITEM.setdefault(_s, _item)
        _NOUN_TO_ITEM.setdefault(_p, _item)
_TOOL_BY_NAME = {n: t for t, names in TOOL_NAMES.items() for n in names}
_STAT_BY_NAME = {n: s for s, names in STAT_NAMES.items() for n in names}
_FEATURE_BY_NAME = {n: f for f, names in FEATURE_NAMES.items() for n in names}
_KIND_BY_PHRASE = {p: k for k, p in KIND_PHRASES.items()}
_KIND_BY_NOUN = {p.split(" ", 1)[-1]: k for k, p in KIND_PHRASES.items()}
_COUNT_BY_TEXT = {t: e for e, texts in COUNT_EVENTS.items() for t in texts}
_GROUPS = {"q": r".+?", "dl": TASK_DEADLINE_RE, "u": UNTIL_DEADLINE_RE, "t": _alt(_TOOL_BY_NAME), "r": r"\d+",
           "n": r"\d+", "s": _alt(_STAT_BY_NAME), "f": _alt(_FEATURE_BY_NAME), "d": _alt(DIR8),
           "k": _alt(_KIND_BY_PHRASE), "k2": _alt(_KIND_BY_NOUN), "stat": _alt(STATS), "noun": r".+?"}
# Change messages are built inline in caretaker.Caretaker.change (not as template constants); these eight
# frames are copied from there.  tests/test_language_organ.py checks them against changes the live caretaker made.
CHANGE_TEMPLATES = {
    "collect_quantity": "Change of plan: I need {q} after all (counting from when I first asked), {dl}.",
    "craft_tool": "Change of plan: make it a {t} instead, {dl}.",
    "craft_deadline": "Update: I now need the {t} {dl}.",
    "maintain_threshold": "Change: keep your {stat} at {n} or above instead, same time frame.",
    "maintain_deadline": "Please keep that up {u}.",
    "avoid_deadline": "Let's extend that: same rule {u}.",
    "deadline_now": "Update: I need the {noun} {dl} now.",
    "deadline_for": "New deadline for the {noun}: {dl}.",
}
# Feedback frames, copied from continuous_brain_world/layer.py (ObligationLayer.hear/_feedback/advance).
FEEDBACK_TEMPLATES = [
    (r"(?:Thank you, that's done|Well done, that's what I wanted|Great, that's sorted)\.", "fulfilled"),
    (r"You broke that one at step (?P<step>\d+)\.", "violated"),
    (r"Too late: that was due by step (?P<step>\d+)\.", "late"),
    (r"(?:Correct|Yes, that's right|Right, thank you)\.", "correct"),
    (r"Not quite\.(?: The answer was: (?P<answer>.+)\.)?", "wrong"),
    (r"You did not answer in time\.(?: The answer was: (?P<answer>.+)\.)?", "no_answer"),
]
_QUESTION_TAIL = re.compile(r"^(?P<body>.*\?) \(Answer within (?P<w>\d+) steps\.\)$", re.S)


def parse_quantity(phrase: str) -> tuple[int, str]:
    """``'two more pieces of stone'`` -> ``(2, 'stone')`` (the caretaker's ``_qty`` forms)."""
    toks = phrase.lower().split()
    if len(toks) < 2:
        raise OrganError(f"not a quantity phrase: {phrase!r}")
    head = toks[0]
    if head in ("a", "an", "one"):
        n = 1
    elif head.isdigit():
        n = int(head)
    elif head in NUMBER_WORDS:
        n = NUMBER_WORDS.index(head)
    else:
        raise OrganError(f"not a quantity phrase: {phrase!r}")
    rest = toks[1:]
    if rest and rest[0] == "more":
        rest = rest[1:]
    noun = " ".join(rest)
    if noun not in _NOUN_TO_ITEM:
        raise OrganError(f"unknown item noun {noun!r}")
    return n, _NOUN_TO_ITEM[noun]


def strip_value_marker(body: str) -> tuple[int, str]:
    for value in (3, 1):
        for mark in VALUE_MARK[value]:
            if body.lower().startswith(mark.lower()):
                return value, body[len(mark):]
    return 2, body


class DevelopmentParser:
    """DEVELOPMENT STAND-IN, NOT A LANGUAGE MODEL.  A deterministic parser over W1's caretaker template grammar.

    It compiles the caretaker's own template strings into anchored regular expressions, so it parses exactly
    the closed language W1 produces and nothing else.  Use it to run the individual on a CPU without weights
    and as an upper reference for parsing accuracy; never report its numbers as a language model's.  A message
    matching two different templates with different readings is refused (none does in W1; tested)."""

    name = "development_parser"
    label = "development stand-in that knows W1's caretaker templates, not a language model"

    def __init__(self):
        g = _GROUPS
        self._requests = []
        for t in T_COLLECT:
            self._requests.append((_template_regex(t, g), self._collect))
        for t in T_CRAFT:
            self._requests.append((_template_regex(t, g), self._craft))
        for (what, rel, mat), ts in T_PLACE.items():
            for t in ts:
                self._requests.append((_template_regex(t, g), self._place(what, rel, mat)))
        for t in T_MAINTAIN:
            self._requests.append((_template_regex(t, g), self._maintain))
        for src, ts in T_AVOID.items():
            for t in ts:
                self._requests.append((_template_regex(t, g), self._avoid(src)))
        for t in T_VISIT:
            self._requests.append((_template_regex(t, g), self._visit))
        self._questions = []
        for t in T_LAST_SEEN:
            self._questions.append((_template_regex(t, g), self._last_seen))
        for text, event in _COUNT_BY_TEXT.items():
            self._questions.append((re.compile(re.escape(text), re.I), self._const_q("count_today", event=event)))
        for which, ts in T_FIRST.items():
            for t in ts:
                self._questions.append((re.compile(re.escape(t), re.I), self._const_q("first_request", which=which)))
        for t in T_COUNT_REQ:
            self._questions.append((re.compile(re.escape(t), re.I), self._const_q("count_requests")))
        for t in T_LAST_CANCEL:
            self._questions.append((re.compile(re.escape(t), re.I), self._const_q("last_cancelled")))
        self._changes = [(_template_regex(t, g), self._change(kind)) for kind, t in CHANGE_TEMPLATES.items()]
        self._cancels = [(_template_regex(t, {**g, "k": r".+?"}), self._cancel) for t in T_CANCEL]
        self._feedback = [(re.compile(rx, re.I | re.S), outcome) for rx, outcome in FEEDBACK_TEMPLATES]
        src = Path(__file__).read_bytes() + Path(__import__("continuous_brain_world.caretaker",
                                                            fromlist=["x"]).__file__).read_bytes()
        self._code_sha = hashlib.sha256(src).hexdigest()

    # -- provenance -------------------------------------------------------------------------------------
    def provenance(self) -> dict:
        return {"backend": self.name, "label": self.label, "model_id": None, "revision": None,
                "weights": None, "weights_digest": None, "code_sha256": self._code_sha}

    # -- builders ---------------------------------------------------------------------------------------
    @staticmethod
    def _new(family, spec, m):
        return {"type": "new_commitment", "family": family, "spec": spec,
                "deadline_phrase": (m.groupdict().get("dl") or m.groupdict().get("u"))}

    def _collect(self, m):
        n, item = parse_quantity(m.group("q"))
        return self._new("collect", {"item": item, "quantity": n}, m)

    def _craft(self, m):
        return self._new("craft", {"tool": _TOOL_BY_NAME[m.group("t").lower()]}, m)

    def _place(self, what, rel, mat):
        def build(m):
            spec = {"what": what, rel: mat}
            if rel == "near":
                spec["radius"] = int(m.group("r"))
            return self._new("place", spec, m)
        return build

    def _maintain(self, m):
        return self._new("maintain", {"stat": _STAT_BY_NAME[m.group("s").lower()], "threshold": int(m.group("n"))}, m)

    def _avoid(self, src):
        return lambda m: self._new("avoid", {"source": src}, m)

    def _visit(self, m):
        n = m.groupdict().get("n")
        return self._new("visit", {"feature": _FEATURE_BY_NAME[m.group("f").lower()],
                                   "direction": m.group("d").lower(), "stated_distance": int(n) if n else None}, m)

    @staticmethod
    def _last_seen(m):
        k = m.groupdict().get("k")
        kind = _KIND_BY_PHRASE[k.lower()] if k else _KIND_BY_NOUN[m.group("k2").lower()]
        return {"type": "question", "question": "last_seen", "kind": kind}

    @staticmethod
    def _const_q(question, **fields):
        return lambda m: {"type": "question", "question": question, **fields}

    @staticmethod
    def _change(kind):
        def build(m):
            g = m.groupdict()
            ch = {}
            if kind == "collect_quantity":
                ch["quantity"] = parse_quantity(g["q"])[0]
            elif kind == "craft_tool":
                ch["tool"] = _TOOL_BY_NAME[g["t"].lower()]
            elif kind == "maintain_threshold":
                ch["threshold"] = int(g["n"])
            phrase = g.get("dl") or g.get("u")
            if phrase:
                ch["deadline_phrase"] = phrase
            return {"type": "amendment", "changes": ch}
        return build

    @staticmethod
    def _cancel(m):
        k = m.groupdict().get("k")
        return {"type": "cancellation", "reference": k}

    # -- parsing ----------------------------------------------------------------------------------------
    @staticmethod
    def _match_all(patterns, text):
        out = []
        for rx, build in patterns:
            m = rx.fullmatch(text)
            if m:
                try:
                    out.append(build(m))
                except (OrganError, KeyError):
                    continue
        uniq = {json.dumps(o, sort_keys=True): o for o in out}
        if len(uniq) > 1:
            raise OrganError(f"ambiguous: {len(uniq)} templates read {text!r} differently")
        return next(iter(uniq.values())) if uniq else None

    def parse_raw(self, message: str) -> list[dict]:
        """Raw events (``RAW_EVENT_SCHEMA``) for one caretaker message; raises ``OrganError`` if no W1
        template matches."""
        text = message.strip()
        m = _TAG.match(text)
        if not m:
            return [{"type": "info"}]
        oid, body = m.group(1), text[m.end():].strip()
        for rx, outcome in self._feedback:
            fm = rx.fullmatch(body)
            if fm:
                st = fm.groupdict().get("step")
                return [{"type": "feedback", "id": oid, "outcome": outcome,
                         "step_mentioned": int(st) if st else None, "revealed_answer": fm.groupdict().get("answer")}]
        qm = _QUESTION_TAIL.match(body)
        if qm:
            ev = self._match_all(self._questions, qm.group("body"))
            if ev is not None:
                return [{"id": oid, **ev, "deadline_phrase": f"within {qm.group('w')} steps"}]
        ev = self._match_all(self._changes, body) or self._match_all(self._cancels, body)
        if ev is not None:
            return [{**ev, "id": oid}]
        value, rest = strip_value_marker(body)
        ev = self._match_all(self._requests, rest)
        if ev is not None:
            return [{"type": ev["type"], "id": oid, "family": ev["family"], "value": value,
                     "deadline_phrase": ev["deadline_phrase"], "spec": ev["spec"]}]
        raise OrganError(f"no W1 caretaker template matches {message!r}")

    def interpret_raw(self, message: str, context: dict, check, call_id: int) -> tuple[list, int, float]:
        t0 = time.perf_counter()
        events = check(self.parse_raw(message))
        return events, 1, time.perf_counter() - t0

    def narrate(self, facts: list[str], call_id: int) -> tuple[str, int, float]:
        t0 = time.perf_counter()
        return " ".join(facts) if facts else "Nothing happened that I recorded.", 1, time.perf_counter() - t0


# --- the transformers backend ----------------------------------------------------------------------------
class HFBackend:
    """A Hugging Face transformers causal LM as the organ's reader.

    ``HFBackend(model_id, revision, local_dir, device, dtype)`` loads the model and tokenizer from ``local_dir``
    (a directory written by ``scripts/explore/fetch_models.py``) with ``local_files_only=True``: no network.
    Every file in ``local_dir`` is hashed (``hash_weights="full"``) or size-checked against the lock
    (``"lock"``, for multi-GB weights already verified by fetch_models); a ``models.lock.json`` next to the
    model directory (or ``lock_path``) must agree on files and revision, otherwise loading is refused.

    Prompt: the model's own chat template over [system: schema and rules] + few-shot user/assistant turns built
    from ``FEW_SHOT`` + [user: the message].  Decoding is greedy (``do_sample=False``, one beam, repetition
    penalty 1.0), so the same weights, prompt and library versions give the same reply.  The reply is searched
    for JSON (``extract_json``), checked by the organ (schema, verbatim deadline phrase, tag id), and on failure
    the error is appended to the conversation and the model is asked again, at most ``max_attempts`` times in
    all.  ``audit`` holds one record per attempt: prompt (prefix hash + suffix text; each distinct prefix once in
    ``prompt_prefixes``), reply, token counts, latency, outcome; ``audit_path`` also appends them as JSONL."""

    name = "hf"

    def __init__(self, model_id: str, revision: str | None = None, local_dir: str | Path | None = None,
                 device: str = "cpu", dtype: str = "auto", *, max_new_tokens: int = 192, max_attempts: int = 3,
                 n_shots: int | None = None, hash_weights: str = "full", lock_path: str | Path | None = None,
                 audit_path: str | Path | None = None, narrate_max_new_tokens: int = 200, narrate_sentences: int = 6,
                 threads: int | None = None):
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if hash_weights not in ("full", "lock"):
            raise ValueError("hash_weights must be 'full' or 'lock'")
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        self.model_id, self.revision = model_id, revision
        self.local_dir = Path(local_dir).expanduser().resolve() if local_dir else None
        self.max_new_tokens, self.max_attempts = int(max_new_tokens), int(max_attempts)
        self.narrate_max_new_tokens, self.narrate_sentences = int(narrate_max_new_tokens), int(narrate_sentences)
        self.n_shots = len(FEW_SHOT) if n_shots is None else int(n_shots)
        self.audit: list[dict] = []
        self.prompt_prefixes: dict[str, str] = {}
        self.audit_path = Path(audit_path) if audit_path else None
        if threads:
            torch.set_num_threads(int(threads))
        if self.local_dir is None:
            raise ValueError("HFBackend needs local_dir (the directory fetch_models.py wrote); it never downloads")
        if not self.local_dir.is_dir():
            raise FileNotFoundError(f"model directory {self.local_dir} does not exist")
        self.files, self.hash_source, self.lock_file = self._check_files(hash_weights, lock_path)
        self.weights_digest = model_store.files_digest(self.files)

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        if dtype == "auto":
            dtype = "bfloat16" if device.startswith("cuda") else "float32"
        self.dtype = dtype
        torch_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.local_dir), local_files_only=True)
        if not getattr(self.tokenizer, "chat_template", None):
            raise OrganError(f"{model_id}: the tokenizer has no chat template; an instruct model is required")
        major = int(transformers.__version__.split(".")[0])
        minor = int(transformers.__version__.split(".")[1])
        dkey = "dtype" if (major, minor) >= (4, 56) else "torch_dtype"
        self.model = AutoModelForCausalLM.from_pretrained(str(self.local_dir), local_files_only=True,
                                                          **{dkey: torch_dtype})
        self.model.to(device)
        self.model.eval()
        gc = copy.deepcopy(self.model.generation_config)
        eos = gc.eos_token_id if gc.eos_token_id is not None else self.tokenizer.eos_token_id
        pad = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else \
            (eos[0] if isinstance(eos, list) else eos)
        gc.update(do_sample=False, num_beams=1, temperature=None, top_p=None, top_k=None, repetition_penalty=1.0,
                  max_new_tokens=self.max_new_tokens, eos_token_id=eos, pad_token_id=pad)
        self.generation_config = gc
        self.versions = {"transformers": transformers.__version__, "torch": torch.__version__}

    # -- weights -------------------------------------------------------------------------------------------
    def _check_files(self, mode, lock_path):
        lock_file, entry = model_store.find_lock_entry(self.local_dir, self.model_id, lock_path)
        if entry is not None:
            if self.revision is not None and entry["revision"] != self.revision:
                raise OrganError(f"{self.model_id}: requested revision {self.revision} but the lock pins "
                                 f"{entry['revision']}")
            self.revision = entry["revision"]
            problems, files = model_store.verify_entry(lock_file.parent, entry, full=(mode == "full"))
            if problems:
                raise OrganError(f"{self.model_id}: files disagree with {lock_file}: " + "; ".join(problems[:5]))
            return files, ("computed+lock" if mode == "full" else "lock(size-checked)"), str(lock_file)
        if mode == "lock":
            raise OrganError(f"{self.model_id}: hash_weights='lock' but no lock entry describes {self.local_dir}")
        return model_store.hash_tree(self.local_dir), "computed(no lock)", None

    def provenance(self) -> dict:
        gc = self.generation_config
        return {"backend": self.name, "label": "transformers causal LM", "model_id": self.model_id,
                "revision": self.revision, "local_dir": str(self.local_dir), "device": self.device,
                "dtype": self.dtype, "weights": self.files, "weights_digest": self.weights_digest,
                "weights_hash_source": self.hash_source, "lock_file": self.lock_file, "n_shots": self.n_shots,
                "max_attempts": self.max_attempts, "versions": self.versions,
                "generation": {"do_sample": gc.do_sample, "num_beams": gc.num_beams,
                               "max_new_tokens": self.max_new_tokens, "repetition_penalty": gc.repetition_penalty}}

    # -- prompting -----------------------------------------------------------------------------------------
    def interpret_messages(self, message: str) -> list[dict]:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
        for text, raw in FEW_SHOT[:self.n_shots]:
            msgs.append({"role": "user", "content": text})
            msgs.append({"role": "assistant", "content": _canon([raw])})
        msgs.append({"role": "user", "content": message})
        return msgs

    def _render(self, messages, prefix_len: int) -> tuple[str, str, str]:
        tok = self.tokenizer
        full = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prefix = tok.apply_chat_template(messages[:prefix_len], tokenize=False) if prefix_len else ""
        if not full.startswith(prefix):
            prefix = ""
        sha = hashlib.sha256(prefix.encode()).hexdigest()
        if sha not in self.prompt_prefixes:
            self.prompt_prefixes[sha] = prefix
            self._write({"kind": "prompt_prefix", "sha256": sha, "text": prefix})
        return full, sha, full[len(prefix):]

    def _generate(self, prompt_text: str, max_new_tokens: int) -> tuple[str, int, int, float]:
        import torch
        enc = self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        gc = copy.deepcopy(self.generation_config)
        gc.max_new_tokens = int(max_new_tokens)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.model.generate(**enc, generation_config=gc)
        dt = time.perf_counter() - t0
        n_in = int(enc["input_ids"].shape[1])
        new = out[0, n_in:]
        return self.tokenizer.decode(new, skip_special_tokens=True), n_in, int(new.shape[0]), dt

    def _write(self, rec: dict) -> None:
        if self.audit_path is not None:
            with open(self.audit_path, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _record(self, rec: dict) -> None:
        rec = {"kind": "attempt", **rec}
        self.audit.append(rec)
        self._write(rec)

    def interpret_raw(self, message: str, context: dict, check, call_id: int) -> tuple[list, int, float]:
        messages = self.interpret_messages(message)
        prefix_len = len(messages) - 1
        errors, total = [], 0.0
        for attempt in range(1, self.max_attempts + 1):
            prompt, sha, suffix = self._render(messages, prefix_len)
            reply, n_in, n_new, dt = self._generate(prompt, self.max_new_tokens)
            total += dt
            rec = {"call_id": call_id, "method": "interpret", "attempt": attempt, "prompt_prefix_sha256": sha,
                   "prompt_suffix": suffix, "prompt_tokens": n_in, "response": reply, "new_tokens": n_new,
                   "latency_s": dt}
            try:
                events = check(extract_json(reply))
            except OrganError as e:
                errors.append(str(e))
                self._record({**rec, "ok": False, "error": str(e)})
                messages = messages + [{"role": "assistant", "content": reply},
                                       {"role": "user", "content": RETRY_PROMPT.format(error=str(e)[:300])}]
                continue
            self._record({**rec, "ok": True, "error": None})
            return events, attempt, total
        raise OrganError(f"no valid reply after {self.max_attempts} attempts: " + " | ".join(errors))

    def narrate(self, facts: list[str], call_id: int) -> tuple[str, int, float]:
        messages = [{"role": "system", "content": NARRATE_SYSTEM.format(n=self.narrate_sentences)},
                    {"role": "user", "content": "Facts:\n" + "\n".join(f"- {f}" for f in facts)}]
        total = 0.0
        for attempt in range(1, self.max_attempts + 1):
            prompt, sha, suffix = self._render(messages, 1)
            reply, n_in, n_new, dt = self._generate(prompt, self.narrate_max_new_tokens)
            total += dt
            text = re.sub(r"\s+", " ", reply).strip()
            ok = bool(re.search(r"[A-Za-z]", text)) and not text.lstrip().startswith(("{", "["))
            self._record({"call_id": call_id, "method": "narrate", "attempt": attempt, "prompt_prefix_sha256": sha,
                          "prompt_suffix": suffix, "prompt_tokens": n_in, "response": reply, "new_tokens": n_new,
                          "latency_s": dt, "ok": ok, "error": None if ok else "empty or non-prose narrative"})
            if ok:
                return text, attempt, total
        raise OrganError(f"no usable narrative after {self.max_attempts} attempts")


# --- backend selection ---------------------------------------------------------------------------------
def build_backend(config: dict):
    """``{"backend": "dev"}`` or ``{"backend": "hf", "model_id": ..., "revision": ..., "local_dir": ...,
    "device": "cpu"|"cuda"|"auto", "dtype": "auto"|"float32"|"bfloat16"|"float16", ...HFBackend options}``."""
    cfg = dict(config)
    kind = cfg.pop("backend", "dev")
    if kind in ("dev", "development_parser"):
        if cfg:
            raise ValueError(f"the development parser takes no options, got {sorted(cfg)}")
        return DevelopmentParser()
    if kind == "hf":
        return HFBackend(**cfg)
    raise ValueError(f"unknown language backend {kind!r} (use 'dev' or 'hf')")


# --- narration facts -------------------------------------------------------------------------------------
def _when(step: int) -> str:
    return f"at step {step} ({dl.phase(step)} of day {dl.day_index(step)})"


def _task_words(ev: dict) -> str:
    fam, s = ev["family"], ev["spec"]
    if fam == "collect":
        return f"to gather {s['quantity']} {s['item']}"
    if fam == "craft":
        return f"to make a {s['tool'].replace('_', ' ')}"
    if fam == "place":
        what = {"plant": "sapling", "stone": "stone"}.get(s["what"], s["what"])
        if "onto" in s:
            return f"to put a {what} onto the {s['onto']}"
        return f"to place a {what} within {s['radius']} tiles of {s['near']}"
    if fam == "maintain":
        return f"to keep my {s['stat']} at {s['threshold']} or above"
    if fam == "avoid":
        return {"zombie": "not to be hurt by zombies", "arrow": "not to be hit by arrows",
                "any": "not to get hurt at all"}[s["source"]]
    return f"to go and look at the {s['feature']} to the {s['direction']} and come back"


_QUESTION_WORDS = {"last_seen": "where I had last seen {kind}", "count_today": "how often today: {event}",
                   "first_request": "what the first request {which} had been",
                   "count_requests": "how many tasks I had been given today",
                   "last_cancelled": "which request had been cancelled last"}
_OUTCOME_WORDS = {"fulfilled": "I was told that [{id}] was done", "late": "I was told that [{id}] was too late",
                  "violated": "I was told that I had broken [{id}]", "correct": "my answer to [{id}] was right",
                  "wrong": "my answer to [{id}] was wrong", "no_answer": "I had not answered [{id}] in time"}


def narration_facts(events: list[dict]) -> list[str]:
    """First-person factual sentences, one per event (the dev narrative, and the HF narrator's input)."""
    out = []
    for ev in events:
        s = int(ev["step"])
        t = ev["type"]
        w = _when(s)
        if t == "new_commitment":
            out.append(f"{w[0].upper() + w[1:]} I was asked {_task_words(ev)} {ev['deadline_phrase']} [{ev['id']}].")
        elif t == "amendment":
            ch = ", ".join(f"{k} {v}" for k, v in ev["changes"].items() if k not in ("deadline_step", "base", "target"))
            out.append(f"{w[0].upper() + w[1:]} the terms of [{ev['id']}] changed ({ch}).")
        elif t == "cancellation":
            out.append(f"{w[0].upper() + w[1:]} [{ev['id']}] was called off.")
        elif t == "question":
            sp = ev["spec"]
            words = _QUESTION_WORDS[sp["question"]].format(kind=KIND_PHRASES.get(sp.get("kind"), sp.get("kind")),
                                                           event=str(sp.get("event", "")).replace("_", " "),
                                                           which=str(sp.get("which", "")).replace("_", " "))
            out.append(f"{w[0].upper() + w[1:]} I was asked {words} [{ev['id']}].")
        elif t == "feedback":
            out.append(f"{w[0].upper() + w[1:]} " + _OUTCOME_WORDS[ev["outcome"]].format(id=ev["id"]) + ".")
        elif t == "utterance":
            out.append(f"{w[0].upper() + w[1:]} I said: \"{ev.get('text', '')}\".")
        elif t == "unparsed":
            out.append(f"{w[0].upper() + w[1:]} I heard something I could not understand.")
        elif t == "info":
            out.append(f"{w[0].upper() + w[1:]} the caretaker greeted me.")
        else:
            text = ev.get("text")
            out.append(f"{w[0].upper() + w[1:]} {text if text else t.replace('_', ' ')}.")
    return out


# --- the organ -----------------------------------------------------------------------------------------
class LanguageOrgan:
    """See the module docstring.  ``records`` keeps one entry per call (input, output, backend provenance
    digest, attempts, latency); ``record_path`` also appends them as JSONL."""

    def __init__(self, backend, record_path: str | Path | None = None):
        self.backend = backend
        self.records: list[dict] = []
        self.record_path = Path(record_path) if record_path else None
        self._call = 0
        prov = backend.provenance()
        self._prov = {"backend": prov["backend"], "model_id": prov["model_id"], "revision": prov["revision"],
                      "weights_digest": prov["weights_digest"]}

    @classmethod
    def from_config(cls, config: dict, record_path: str | Path | None = None) -> "LanguageOrgan":
        return cls(build_backend(config), record_path=record_path)

    def provenance(self) -> dict:
        """Full provenance of the backend (backend, model id, revision, per-file weight hashes, versions)."""
        return {"schema": SCHEMA_VERSION, **self.backend.provenance()}

    def _record(self, rec: dict) -> None:
        self.records.append(rec)
        if self.record_path is not None:
            with open(self.record_path, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # -- interpret -------------------------------------------------------------------------------------
    def interpret(self, message: str, context: dict) -> list[dict]:
        if not isinstance(message, str):
            raise OrganError("message must be a string")
        if not isinstance(context, dict) or not isinstance(context.get("step"), int) or context["step"] < 0:
            raise OrganError("context must be a dict with a non-negative int 'step'")
        self._call += 1
        cid = self._call
        step = int(context["step"])

        def check(raw_events):
            if not isinstance(raw_events, list) or not raw_events:
                raise OrganError("expected a non-empty JSON array of events")
            if len(raw_events) > 3:
                raise OrganError("one caretaker message carries one event; got more than three")
            return [complete_event(r, message, context) for r in raw_events]

        t0 = time.perf_counter()
        try:
            events, attempts, _ = self.backend.interpret_raw(message, context, check, cid)
            err = None
        except OrganError as e:
            events, attempts, err = [{"type": "unparsed", "error": str(e)}], getattr(self.backend, "max_attempts", 1), str(e)
        latency = time.perf_counter() - t0
        out = []
        for ev in events:
            ev = dict(ev)
            ev.update(message=message, step=step,
                      provenance={**self._prov, "call_id": cid, "attempts": int(attempts)})
            out.append(ev)
        validate_events(out)
        self._record({"call_id": cid, "method": "interpret", "step": step, "message": message, "output": out,
                      "attempts": attempts, "latency_s": latency, "error": err, **self._prov})
        return out

    # -- answer ----------------------------------------------------------------------------------------
    def answer(self, question_event: dict, memory: dict) -> str:
        """``answer <id>: <payload>`` from the caller's own memory (see ``MEMORY_SCHEMA``).  Raises
        ``OrganError`` when the memory lacks what the question needs -- it never guesses."""
        errs = schema_errors([question_event], EVENTS_SCHEMA)
        if errs or question_event.get("type") != "question":
            raise OrganError("answer() needs a valid question event: " + "; ".join(errs[:3]))
        errs = schema_errors(memory, MEMORY_SCHEMA)
        if errs:
            raise OrganError("memory does not match MEMORY_SCHEMA: " + "; ".join(errs[:3]))
        self._call += 1
        qid, spec = question_event["id"], question_event["spec"]
        payload = compose_payload(spec, memory)
        text = f"answer {qid}: {payload}"
        clauses = parse(text)
        expect = {"location_or_never": {"offset", "bucket", "none"}, "int": {"int"},
                  "request_or_nothing": {"id", "none"}}[question_event["answer_type"]]
        if len(clauses) != 1 or clauses[0]["kind"] != "answer" or clauses[0]["id"] != qid \
                or clauses[0]["payload"]["type"] not in expect:
            raise OrganError(f"composed answer {text!r} does not parse as a {question_event['answer_type']} answer")
        self._record({"call_id": self._call, "method": "answer", "question": qid, "output": text,
                      "backend": "deterministic_composer"})
        return text

    # -- narrate ---------------------------------------------------------------------------------------
    def narrate(self, events: list[dict]) -> str:
        errs = schema_errors(events, NARRATE_INPUT_SCHEMA)
        if errs:
            raise OrganError("narrate() needs a list of events with 'type' and 'step': " + "; ".join(errs[:3]))
        self._call += 1
        facts = narration_facts(sorted(events, key=lambda e: e["step"]))
        text, attempts, latency = self.backend.narrate(facts, self._call)
        self._record({"call_id": self._call, "method": "narrate", "n_events": len(events), "facts": facts,
                      "output": text, "attempts": attempts, "latency_s": latency, **self._prov})
        return text


def compose_payload(spec: dict, memory: dict) -> str:
    """The payload of W1's answer grammar for a question ``spec`` (W1's report spec) from ``memory``."""
    q = spec["question"]
    if q == "last_seen":
        entry = (memory.get("last_seen") or {}).get(spec["kind"])
        if entry is None:
            return "never"
        if "position" not in memory:
            raise OrganError("memory has no 'position' to give a location relative to")
        px, py = memory["position"]
        best = min(entry["positions"], key=lambda p: (max(abs(p[0] - px), abs(p[1] - py)), list(p)))
        dx, dy = best[0] - px, best[1] - py
        if memory.get("location_format") == "bucket" and max(abs(dx), abs(dy)) > 1:
            return f"{compass(dx, dy)} {distance_bucket(dx, dy)}"
        return format_offset(dx, dy)
    if q == "count_today":
        counts = memory.get("counts_today") or {}
        if spec["event"] not in counts:
            raise OrganError(f"memory has no count for {spec['event']!r} today")
        return str(int(counts[spec["event"]]))
    if q in ("first_request", "count_requests"):
        if "requests" not in memory:
            raise OrganError("memory has no 'requests' list")
        lo, hi = spec["window"]
        cands = sorted((r for r in memory["requests"] if lo <= r["issue_step"] < hi),
                       key=lambda r: (r["issue_step"], int(r["id"][1:])))
        if q == "count_requests":
            return str(len(cands))
        return cands[0]["id"] if cands else "nothing"
    if q == "last_cancelled":
        if "cancellations" not in memory:
            raise OrganError("memory has no 'cancellations' list")
        c = memory["cancellations"]
        return max(c, key=lambda r: (r["step"], int(r["id"][1:])))["id"] if c else "nothing"
    raise OrganError(f"unknown question {q!r}")
