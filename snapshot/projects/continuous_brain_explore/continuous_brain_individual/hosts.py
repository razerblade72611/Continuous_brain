"""The individual's faculties as runtime components over the canonical store (exploration lane).

The same invariant as I1's stateful hosts (``continuous_brain_unified.integration.hosts``): each faculty keeps its
persistent state in the blob channel ``<slot>.state``, reads it as of the start of the tick (a delayed read, so a
store intervention is what it sees), and writes the new state back.  Nothing about the individual is kept in
Python between ticks; the organs are fixed functions with pinned weights.

Tick order (by data dependency; the runtime's scheduler derives it):

    driver.frame, driver.body, driver.messages          written by the life loop from W1 (body sense only)
    perception     vision organ: frame -> perceived tiles          (or oracle tiles, in the oracle ablation)
    episodic_memory  integrates the tiles into the remembered world map, last-seen log, day snapshots
    goal_genesis   language organ: messages -> events -> the commitment ledger
    self_state     self-model: per-family success rates from feedback, plus vitals
    metacognition  speech: answers pending questions from its own memory, acknowledges requests
    skills         belief-driven skill routines -> the action
    autobiography  the day's events, narrated by the language organ at dawn; expected-vs-actual discrepancy

Provenance: every host is ``TEST_FIXTURE``-kind with ``lane: exploration`` (engineering only, never scientific);
the runtime marks all of its output non-scientific.
"""
from __future__ import annotations

import json

import numpy as np

from continuous_brain_unified.channels import ChannelDecl
from continuous_brain_unified.components import Component, ComponentProvenance, ProvenanceKind, TickContext
from continuous_brain_organs.language import OrganError
from continuous_brain_world import daylight as dl

from . import codec, ledger, memory as mem
from .body import VITALS, read_body
from .skills import ACTIONS, SkillLibrary

ROWS, COLS = mem.ROWS, mem.COLS
OPTION_CODES = {"routine": 0, "explore": 1, "sleeping": 2, "eat": 3}
DIARY_DAYS, LIVES_KEPT = 120, 300     # its diary keeps the most recent days; its record, the most recent lives


def state_channel(slot: str) -> str:
    return f"{slot}.state"


def _blob_decl(name, slot, desc):
    return ChannelDecl(name, slot, None, "uint8", intervenable=True, description=desc)


def _text(arr) -> str:
    return np.asarray(arr, dtype=np.uint8).tobytes().decode()


def _json(arr):
    raw = np.asarray(arr, dtype=np.uint8).tobytes()
    return json.loads(raw.decode()) if raw else None


def _bytes(text: str) -> np.ndarray:
    return np.frombuffer(text.encode(), dtype=np.uint8).copy()


class _Host(Component):
    mechanism: str = ""

    def __init__(self, organ_provenance: dict | None = None):
        self.name = f"individual:{self.mechanism}"
        self._organ = organ_provenance or {}

    def provenance(self) -> ComponentProvenance:
        return ComponentProvenance(ProvenanceKind.TEST_FIXTURE, {
            "lane": "exploration", "engineering_only": True, "mechanism": self.mechanism, "organ": self._organ,
            "note": "W1 individual v1; store-held state; never scientific"})


# ---------------------------------------------------------------------------------------------------- perception
class PerceptionHost(_Host):
    slot, mechanism = "perception", "vision-organ"

    def __init__(self, vision, oracle: bool = False):
        prov = {"weights_sha256": getattr(vision, "weights_sha256", None), "oracle": oracle}
        super().__init__(prov)
        self.vision, self.oracle = vision, oracle
        self.reads = ("driver.frame", "driver.oracle_tiles")
        self.writes = (_blob_decl("perception.tiles", self.slot, "perceived local tile map (codec)"),
                       ChannelDecl("perception.features", self.slot, (2, ROWS, COLS), "float64",
                                   description="perceived material and object class per local tile"))

    def step(self, ctx: TickContext):
        if self.oracle:
            tiles = codec.unblob(ctx.inputs["driver.oracle_tiles"])
        else:
            tiles = self.vision.perceive(np.asarray(ctx.inputs["driver.frame"], dtype=np.uint8))
        tiles = {k: np.asarray(v) for k, v in tiles.items()}
        feats = np.stack([tiles["material"], tiles["object"]]).astype(np.float64)
        return {"perception.tiles": codec.blob(tiles), "perception.features": feats}


# ---------------------------------------------------------------------------------------------- episodic memory
class MemoryHost(_Host):
    slot, mechanism = "episodic_memory", "world-map-and-episodes"

    def __init__(self, mode: str = "full"):
        super().__init__({"mode": mode})
        self.mode = mode
        self.reads = ("perception.tiles", "driver.body")
        self.reads_delayed = (state_channel(self.slot), "skills.action")      # its own previous action
        self.writes = (_blob_decl(state_channel(self.slot), self.slot, "the individual's episodic memory (codec)"),
                       ChannelDecl("episodic_memory.slots", self.slot, (4,), "float64",
                                   description="[known fraction, kinds seen, objects in view, steps lived]"))

    def step(self, ctx):
        state = codec.unblob(ctx.delayed[state_channel(self.slot)])
        body = read_body(ctx.inputs["driver.body"])
        if state is not None and state.get("life", 0) != body.get("life", 0):
            state = None                                        # a new world: nothing it saw holds here
        tiles = codec.unblob(ctx.inputs["perception.tiles"])
        prev = np.asarray(ctx.delayed["skills.action"])
        last_action = int(prev[0]) if state is not None else None
        new = mem.integrate(state, tiles, body, int(body["step"]), self.mode, last_action)
        if "life" in body:
            new["life"] = int(body["life"])
        return {state_channel(self.slot): codec.blob(new), "episodic_memory.slots": mem.summary_vector(new)}


# ------------------------------------------------------------------------------------------- commitment ledger
class LedgerHost(_Host):
    slot, mechanism = "goal_genesis", "commitment-ledger"

    def __init__(self, language, mode: str = "organ"):
        super().__init__({**language.provenance(), "mode": mode})
        self.language, self.mode = language, mode
        self.reads = ("driver.messages", "driver.body")
        self.reads_delayed = (state_channel(self.slot),)
        self.writes = (_blob_decl(state_channel(self.slot), self.slot, "the commitment ledger (codec)"),
                       _blob_decl("goal_genesis.events", self.slot, "this tick's language events (JSON)"),
                       ChannelDecl("goal_genesis.active_goal", self.slot, (len(ledger.TASK_FAMILIES) + 1,),
                                   "float64", description="open commitments per family, then open questions"))

    def step(self, ctx):
        state = codec.unblob(ctx.delayed[state_channel(self.slot)]) or ledger.empty()
        body = read_body(ctx.inputs["driver.body"])
        if state.get("life", 0) != body.get("life", 0):        # a new world: a new caretaker's requests
            state = {**ledger.empty(), "organ_calls": state["organ_calls"]}
        step = int(body["step"])
        messages = _json(ctx.inputs["driver.messages"]) or []
        events = []
        if self.mode == "organ":
            heard = state.setdefault("heard", {})
            self.language._call = int(state["organ_calls"])          # the organ's counter lives in the store
            for m in messages:
                context = {"step": step, "inventory": body["inventory"], "achievements": body["achievements"],
                           "player_pos": body["player_pos"], "commitments": heard}
                evs = self.language.interpret(m, context)
                for e in evs:
                    if e["type"] == "new_commitment":
                        heard[e["id"]] = e
                events.extend(evs)
            state["organ_calls"] = int(self.language._call)
            del self.language.records[:]                            # audit is per call; not individual state
        new = ledger.apply_events(state, events, step)
        new["heard"] = state.get("heard", {})
        if "life" in body:
            new["life"] = int(body["life"])
        return {state_channel(self.slot): codec.blob(new),
                "goal_genesis.events": _bytes(json.dumps(events, sort_keys=True, separators=(",", ":"))),
                "goal_genesis.active_goal": np.array(ledger.summary_vector(new), dtype=np.float64)}


# --------------------------------------------------------------------------------------------------- self-model
class SelfModelHost(_Host):
    slot, mechanism = "self_state", "self-model"
    FAMILIES = ledger.TASK_FAMILIES + ("report",)

    def __init__(self):
        super().__init__()
        self.reads = ("goal_genesis.events", "goal_genesis.state", "driver.body")
        self.reads_delayed = (state_channel(self.slot),)
        self.writes = (_blob_decl(state_channel(self.slot), self.slot, "the self-model (codec)"),
                       ChannelDecl("self_state.vector", self.slot, (len(self.FAMILIES) + len(VITALS),), "float64",
                                   description="believed success rate per family (Beta(1,1) mean), then vitals/9"))

    def step(self, ctx):
        state = codec.unblob(ctx.delayed[state_channel(self.slot)]) or {f: [0, 0] for f in self.FAMILIES}
        events = _json(ctx.inputs["goal_genesis.events"]) or []
        led = codec.unblob(ctx.inputs["goal_genesis.state"]) or ledger.empty()
        body = read_body(ctx.inputs["driver.body"])
        state = {k: list(v) for k, v in state.items()}
        for e in events:
            if e["type"] != "feedback":
                continue
            c = led["commitments"].get(e["id"])
            fam = c["family"] if c is not None else ("report" if e["id"] in led["questions"] else None)
            if fam is None:
                continue
            ok = e["outcome"] in ("fulfilled", "correct")
            state[fam][1] += 1
            state[fam][0] += int(ok)
        rates = [(state[f][0] + 1) / (state[f][1] + 2) for f in self.FAMILIES]
        vit = [body["inventory"][k] / 9.0 for k in VITALS]
        return {state_channel(self.slot): codec.blob(state), "self_state.vector": np.array(rates + vit)}


# ------------------------------------------------------------------------------------------------------- speech
class SpeechHost(_Host):
    slot, mechanism = "metacognition", "speech-from-memory"

    def __init__(self, language, mode: str = "organ"):
        super().__init__({**language.provenance(), "mode": mode})
        self.language, self.mode = language, mode
        self.reads = ("goal_genesis.state", "goal_genesis.events", "episodic_memory.state", "driver.body")
        self.reads_delayed = (state_channel(self.slot),)
        self.writes = (_blob_decl(state_channel(self.slot), self.slot, "what it has said (codec)"),
                       _blob_decl("metacognition.utterance", self.slot, "this tick's utterance (text)"),
                       ChannelDecl("metacognition.confidence", self.slot, (1,), "float64",
                                   description="fraction of this tick's questions it could answer from memory"))

    def step(self, ctx):
        state = codec.unblob(ctx.delayed[state_channel(self.slot)]) or {"answered": set(), "acked": set(),
                                                                         "unanswerable": []}
        life = read_body(ctx.inputs["driver.body"]).get("life", 0)
        if state.get("life", 0) != life:                        # a new world: ids start over
            state = {"answered": set(), "acked": set(), "unanswerable": []}
        state = {"answered": set(state["answered"]), "acked": set(state["acked"]),
                 "unanswerable": list(state["unanswerable"]), **({"life": life} if life else {})}
        clauses, asked, ok = [], 0, 0
        if self.mode == "organ":
            led = codec.unblob(ctx.inputs["goal_genesis.state"]) or ledger.empty()
            memory = codec.unblob(ctx.inputs["episodic_memory.state"]) or mem.empty()
            body = read_body(ctx.inputs["driver.body"])
            step = int(body["step"])
            recall = {"position": list(body["player_pos"]),
                      "last_seen": {k: {"step": int(v["step"]), "positions": [list(p) for p in v["positions"]]}
                                    for k, v in memory["last_seen"].items()},
                      "counts_today": mem.counts_today(memory, body, step),
                      "requests": [dict(r) for r in led["requests"]],
                      "cancellations": [dict(c) for c in led["cancellations"]], "location_format": "offset"}
            for q in ledger.pending_questions(led):
                if q["id"] in state["answered"]:
                    continue
                asked += 1
                try:
                    clauses.append(self.language.answer(q, recall))
                    ok += 1
                except OrganError as e:
                    state["unanswerable"].append({"id": q["id"], "step": step, "why": str(e)[:200]})
                state["answered"].add(q["id"])
            for e in _json(ctx.inputs["goal_genesis.events"]) or []:
                if e["type"] == "new_commitment" and e["id"] not in state["acked"]:
                    clauses.append(f"ack {e['id']}")
                    state["acked"].add(e["id"])
            del self.language.records[:]
        conf = ok / asked if asked else 1.0
        return {state_channel(self.slot): codec.blob(state), "metacognition.utterance": _bytes("; ".join(clauses)),
                "metacognition.confidence": np.array([conf])}


# ------------------------------------------------------------------------------------------------------- skills
class SkillHost(_Host):
    slot, mechanism = "skills", "belief-driven-skill-library"

    def __init__(self, seed: int = 0):
        super().__init__({"library": "W1 privileged reference routines over the individual's beliefs"})
        self.seed = seed
        self.reads = ("episodic_memory.state", "goal_genesis.state", "driver.body")
        self.reads_delayed = (state_channel(self.slot),)
        self.writes = (_blob_decl(state_channel(self.slot), self.slot, "skill routines' state (codec)"),
                       ChannelDecl("skills.selected_option", self.slot, (1,), "float64",
                                   description="0 routine, 1 explore, 2 sleeping, 3 opportunistic eating"),
                       ChannelDecl("skills.action", self.slot, (1,), "int64", neutral=0,
                                   description="the Crafter action index chosen this tick"))

    def step(self, ctx):
        state = codec.unblob(ctx.delayed[state_channel(self.slot)])
        body = read_body(ctx.inputs["driver.body"])
        if state is not None and state.get("life", 0) != body.get("life", 0):
            state = None                                        # a new world: its plans there are gone
        lib = SkillLibrary.from_state(state, seed=self.seed)
        memory = codec.unblob(ctx.inputs["episodic_memory.state"]) or mem.empty()
        led = codec.unblob(ctx.inputs["goal_genesis.state"]) or ledger.empty()
        step = int(body["step"])
        name, option = lib.decide(memory, body, step, ledger.active(led, step))
        code = OPTION_CODES.get(option.split(":")[0], 0)
        new = lib.to_state()
        if "life" in body:
            new["life"] = int(body["life"])
        return {state_channel(self.slot): codec.blob(new), "skills.selected_option": np.array([code], float),
                "skills.action": np.array([ACTIONS[name]], dtype=np.int64)}


# ------------------------------------------------------------------------------------------------ autobiography
class AutobiographyHost(_Host):
    slot, mechanism = "autobiography", "narrated-days"

    def __init__(self, language):
        super().__init__(language.provenance())
        self.language = language
        self.reads = ("goal_genesis.events", "goal_genesis.state", "self_state.vector", "driver.body")
        self.reads_delayed = (state_channel(self.slot),)
        self.writes = (_blob_decl(state_channel(self.slot), self.slot, "days lived and narrated (codec)"),
                       ChannelDecl("autobiography.record", self.slot, (3,), "float64",
                                   description="[days narrated, events today, commitments closed]"),
                       ChannelDecl("autobiography.discrepancy", self.slot, (1,), "float64",
                                   description="fulfilled minus self-predicted fulfilments, over closed commitments"))

    def step(self, ctx):
        state = codec.unblob(ctx.delayed[state_channel(self.slot)]) or {"day": 0, "today": [], "days": [],
                                                                         "last_ach": None, "expected": 0.0,
                                                                         "actual": 0, "closed": 0,
                                                                         "organ_calls": 0}
        body = read_body(ctx.inputs["driver.body"])
        step = int(body["step"])
        events = _json(ctx.inputs["goal_genesis.events"]) or []
        led = codec.unblob(ctx.inputs["goal_genesis.state"]) or ledger.empty()
        selfv = np.asarray(ctx.inputs["self_state.vector"])
        fams = list(SelfModelHost.FAMILIES)
        lifelong = "life" in body
        life = int(body.get("life", 0))
        if lifelong and state.get("life", 0) != life:          # it wakes in a new world, the same individual
            state = {**state, "life": life, "day": dl.day_index(step), "last_ach": None,
                     "today": [{"type": "birth", "step": step,
                                "text": f"I woke up in a new world; this is my life number {life + 1}"}]}
        today = list(state["today"])
        for e in events:
            if e["type"] in ("new_commitment", "amendment", "cancellation", "question", "feedback", "unparsed"):
                today.append({k: e[k] for k in e if k != "provenance"})
            if e["type"] == "feedback" and e["id"] in led["commitments"]:
                fam = led["commitments"][e["id"]]["family"]
                state["expected"] = float(state["expected"]) + float(selfv[fams.index(fam)])
                state["actual"] = int(state["actual"]) + int(e["outcome"] == "fulfilled")
                state["closed"] = int(state["closed"]) + 1
        ach = body["achievements"]
        if state["last_ach"] is not None:
            for k, v in sorted(ach.items()):
                if v > state["last_ach"].get(k, 0) and state["last_ach"].get(k, 0) == 0:
                    today.append({"type": "milestone", "step": step,
                                  "text": f"I managed to {k.replace('_', ' ')} for the first time"})
        day = dl.day_index(step)
        days = list(state["days"])
        lives = list(state.get("lives", []))
        dying = bool(body.get("dying", False))
        if dying:
            inv = body["inventory"]
            felt = [w for k, w in (("drink", "no water left"), ("food", "no food left"), ("energy", "no strength left"))
                    if inv[k] == 0]
            today.append({"type": "death", "step": step,
                          "text": "my life ended" + (" with " + " and ".join(felt) if felt else " after I was hurt")})
            lives.append({"life": life, "steps": step, "days": day + 1,
                          "done": sum(1 for v in ach.values() if v), "felt": felt})
        if (day != state["day"] or dying) and today:
            self.language._call = int(state["organ_calls"])
            days.append({"day": int(state["day"]), "text": self.language.narrate(today), "events": len(today),
                         **({"life": life} if lifelong else {})})
            state["organ_calls"] = int(self.language._call)
            del self.language.records[:]
            today = []
        new = {**state, "day": day, "today": today, "days": days[-DIARY_DAYS:], "last_ach": dict(ach),
               **({"lives": lives[-LIVES_KEPT:], "life": life} if lifelong else {})}
        rec = np.array([len(days), len(today), new["closed"]], dtype=np.float64)
        disc = np.array([new["actual"] - new["expected"]], dtype=np.float64)
        return {state_channel(self.slot): codec.blob(new), "autobiography.record": rec,
                "autobiography.discrepancy": disc}
