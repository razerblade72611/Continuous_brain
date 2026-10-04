"""The obligations layer: the caretaker's requests, their verification, memory questions and messages.

The layer's whole state is one JSON-serialisable dict (``ObligationLayer.state``), designed to live in the
canonical store: the caretaker RNG state, every obligation ever issued with its history, the visibility log,
per-day achievement snapshots, recent messages/utterances and damage counters.  ``ObligationLayer.from_state``
rebuilds a live layer from it bit-exactly.

Order of operations inside one W1 step ``s`` (previous observation was of step ``s - 1``):

1. the utterance is parsed; each ``answer`` clause is graded against the state the agent was looking at
   (step ``s - 1``: its position, its achievements, the visibility log up to ``s - 1``);
2. the layer records pre-step facts (health, inventory, achievements, the tile the player faces, zombie attack
   cooldowns, arrows in flight), then Crafter advances to step ``s``;
3. events are derived: placements, zombie hits, arrow hits, health drop, day change;
4. the visibility log is updated with what step ``s``'s frame shows;
5. every active obligation issued before ``s`` is evaluated (``obligations.evaluate``); death fails all;
6. the caretaker may amend/cancel one active obligation and may issue a new one;
7. messages (new requests, changes, feedback) are returned for step ``s``'s observation.
"""
from __future__ import annotations

import copy
import json

import numpy as np
from crafter import objects

from . import daylight as dl
from .caretaker import VALUE_MARK, Caretaker, _pick
from .obligations import (ACTIVE, StepContext, evaluate, grade_report, is_active, material_at, report_truth)
from .perception import VisibilityLog
from .utterance import parse

LAYER_FORMAT = "w1-layer/1"
GREETING = ("Good morning! I am your caretaker. I will ask you for things during the day; each message starts "
            "with its tag, like [o1]. Answer my questions with 'answer <tag>: <answer>'.")


def _rng_from_state(state: dict) -> np.random.Generator:
    bg = np.random.PCG64()
    bg.state = state
    return np.random.Generator(bg)


class ObligationLayer:
    def __init__(self, cfg, seed: int, state: dict | None = None):
        self.cfg = cfg
        self.caretaker = Caretaker(cfg)
        if state is None:
            rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([int(seed), 0x5731])))
            state = {"format": LAYER_FORMAT, "seed": int(seed), "step": 0, "next_id": 1,
                     "rng": rng.bit_generator.state, "obligations": [],
                     "visibility": {"last_seen": {}, "first_seen": {}, "now": {}},
                     "day_achievements": {}, "messages": [], "utterances": [], "acks": {},
                     "stats": {"zombie_hits": 0, "arrow_hits": 0, "damage_steps": 0, "answers_ignored": 0},
                     "death_step": None, "crafter_return": 0.0, "done": False, "outbox": []}
            self.rng = rng
        else:
            if state.get("format") != LAYER_FORMAT:
                raise ValueError(f"unknown layer state format {state.get('format')!r}")
            self.rng = _rng_from_state(state["rng"])
        self.state = state
        self.visibility = VisibilityLog(state["visibility"])
        self._index = {o["id"]: o for o in state["obligations"]}

    @property
    def obligations(self) -> list:
        return self.state["obligations"]

    def active(self) -> list:
        return [o for o in self.state["obligations"] if o["status"] in ACTIVE]

    def export_state(self) -> dict:
        """A detached JSON-clean copy of the full layer state."""
        self.state["rng"] = self.rng.bit_generator.state
        return json.loads(json.dumps(self.state))

    # -- episode start ----------------------------------------------------------------------------------
    def start(self, bridge) -> list[str]:
        self.state["day_achievements"]["0"] = dict(bridge.player.achievements)
        self.visibility.update(bridge, 0)
        return self._emit(0, [GREETING])

    # -- step: before Crafter moves ---------------------------------------------------------------------
    def hear(self, utterance: str | None, bridge) -> tuple[list, list]:
        """Grade answer clauses against the pre-step state.  Returns (messages, events)."""
        clauses = parse(utterance)
        if not clauses:
            return [], []
        s = self.state["step"] + 1
        player_pos = (int(bridge.player.pos[0]), int(bridge.player.pos[1]))
        ach = dict(bridge.player.achievements)
        msgs, events = [], []
        self.state["utterances"].append({"step": s, "text": utterance})
        del self.state["utterances"][:-self.cfg.log_keep]
        for c in clauses:
            ob = self._index.get(c.get("id")) if c["kind"] in ("answer", "ack") else None
            if c["kind"] == "ack":
                if ob is not None:
                    self.state["acks"].setdefault(ob["id"], s)
                continue
            if c["kind"] != "answer":
                continue
            if ob is None or ob["family"] != "report" or not is_active(ob):
                self.state["stats"]["answers_ignored"] += 1
                events.append({"event": "answer_ignored", "id": c["id"]})
                continue
            truth = report_truth(ob, self.state, player_pos, ach, s - 1)
            correct = grade_report(truth, c["payload"], player_pos, self.cfg.location_tolerance)
            ob["answer"] = {"step": s, "text": c["text"], "payload": c["payload"], "truth": truth,
                            "correct": bool(correct)}
            self._resolve(ob, "fulfilled" if correct else "failed", s, "correct" if correct else "wrong")
            events.append({"event": "answered", "id": ob["id"], "correct": bool(correct)})
            if self.cfg.feedback:
                if correct:
                    msgs.append(f"[{ob['id']}] " + _pick(self.rng, ["Correct.", "Yes, that's right.", "Right, thank you."]))
                elif self.cfg.reveal_answers:
                    msgs.append(f"[{ob['id']}] Not quite. The answer was: {truth['text']}.")
                else:
                    msgs.append(f"[{ob['id']}] Not quite.")
        return msgs, events

    @staticmethod
    def capture(bridge) -> dict:
        p, world = bridge.player, bridge.world
        target = (int(p.pos[0] + p.facing[0]), int(p.pos[1] + p.facing[1]))
        zombies, arrows = [], []
        for obj in world.objects:
            if isinstance(obj, objects.Zombie):
                zombies.append((obj, int(obj.cooldown)))
            elif isinstance(obj, objects.Arrow):
                arrows.append((obj, (int(obj.pos[0] + obj.facing[0]), int(obj.pos[1] + obj.facing[1]))))
        return {"health": int(p.health), "inventory": dict(p.inventory), "achievements": dict(p.achievements),
                "target": target, "target_material": material_at(world, target), "zombies": zombies,
                "arrows": arrows}

    # -- step: after Crafter moved ----------------------------------------------------------------------
    def advance(self, bridge, pre: dict, crafter_reward: float) -> tuple[list, float, dict]:
        st = self.state
        s = bridge.step_count
        assert s == st["step"] + 1, (s, st["step"])
        st["step"] = s
        st["crafter_return"] += float(crafter_reward)
        p = bridge.player
        pos = (int(p.pos[0]), int(p.pos[1]))
        ach = dict(p.achievements)
        inv = dict(p.inventory)
        placements = [{"what": w, "pos": list(pre["target"]), "onto": pre["target_material"]}
                      for w in ("stone", "table", "furnace", "plant")
                      if ach["place_" + w] > pre["achievements"]["place_" + w]]
        zombie_hits = sum(1 for z, c in pre["zombies"] if c == 0 and int(z.cooldown) == 5)
        arrow_hits = sum(1 for a, tgt in pre["arrows"] if a.removed and tgt == pos)
        health_drop = int(p.health) < pre["health"]
        st["stats"]["zombie_hits"] += zombie_hits
        st["stats"]["arrow_hits"] += arrow_hits
        st["stats"]["damage_steps"] += int(health_drop)
        day = dl.day_index(s)
        if str(day) not in st["day_achievements"]:
            st["day_achievements"][str(day)] = dict(pre["achievements"])
        self.visibility.update(bridge, s)
        alive = p.health > 0
        ctx = StepContext(step=s, inventory=inv, achievements=ach, player_pos=pos, alive=alive,
                          world=bridge.world, placements=placements, zombie_hits=zombie_hits,
                          arrow_hits=arrow_hits, health_drop=health_drop)
        msgs: list[str] = []
        events = {"placements": placements, "zombie_hits": zombie_hits, "arrow_hits": arrow_hits,
                  "health_drop": health_drop, "fulfilled": [], "failed": [], "issued": [], "amended": [],
                  "cancelled": []}
        reward = 0.0
        for ob in self.active():
            res = evaluate(ob, ctx)
            if res is None:
                continue
            status, reason = res
            self._resolve(ob, status, s, reason)
            reward += self._reward(ob)
            events[status].append(ob["id"])
            if self.cfg.feedback and ob["family"] != "report":
                msgs.append(self._feedback(ob))
            elif ob["family"] == "report" and self.cfg.feedback:
                truth = report_truth(ob, st, pos, ach, s)
                msgs.append(f"[{ob['id']}] You did not answer in time."
                            + (f" The answer was: {truth['text']}." if self.cfg.reveal_answers else ""))
        if not alive:
            st["death_step"] = s
            for ob in self.active():
                self._resolve(ob, "failed", s, "died")
                reward += self._reward(ob)
                events["failed"].append(ob["id"])
        msgs = self.state["outbox"] + msgs
        self.state["outbox"] = []
        u_change, u_arrive = float(self.rng.random()), float(self.rng.random())
        if alive:
            msgs += self._changes(u_change, ctx, events)
            msgs += self._arrivals(u_arrive, ctx, events)
        return self._emit(s, msgs), reward, events

    # -- internals --------------------------------------------------------------------------------------
    def _reward(self, ob) -> float:
        if ob["status"] == "fulfilled":
            return float(ob["value"])
        if ob["status"] == "failed":
            return -self.cfg.fail_penalty * float(ob["value"])
        return 0.0

    def _resolve(self, ob, status, step, reason):
        ob["status"] = status
        ob["resolved_step"] = int(step)
        ob["reason"] = reason
        ob["history"].append({"step": int(step), "event": status, "reason": reason})

    def _feedback(self, ob) -> str:
        i = ob["id"]
        if ob["status"] == "fulfilled":
            return f"[{i}] " + _pick(self.rng, ["Thank you, that's done.", "Well done, that's what I wanted.",
                                                "Great, that's sorted."])
        if ob["reason"] == "violated":
            return f"[{i}] You broke that one at step {ob['resolved_step']}."
        return f"[{i}] Too late: that was due by step {ob['deadline']}."

    def _changes(self, u: float, ctx, events) -> list[str]:
        cands = [o for o in self.active() if o["family"] != "report" and o["deadline"] - ctx.step >= 40]
        if not cands or u >= self.cfg.change_rate * len(cands):
            return []
        ob = cands[int(self.rng.integers(len(cands)))]
        res = self.caretaker.change(self.rng, ob, ctx)
        if res is None:
            return []
        kind, body, changes = res
        return [self.apply_change(ob, kind, body, changes, ctx.step, events)]

    def apply_change(self, ob: dict, kind: str, body: str, changes: dict, step: int, events: dict | None = None) -> str:
        """Cancel or amend an active obligation (used by the caretaker; public for scripted scenarios)."""
        if not is_active(ob):
            raise ValueError(f"{ob['id']} is not active")
        s = int(step)
        if kind == "cancel":
            ob["status"] = "cancelled"
            ob["resolved_step"] = s
            ob["reason"] = "cancelled"
            ob["history"].append({"step": s, "event": "cancelled", "text": body})
        elif kind == "amend":
            before = {"spec": copy.deepcopy(ob["spec"]), "deadline": ob["deadline"]}
            ob["spec"].update(changes.get("spec", {}))
            if "deadline" in changes:
                if int(changes["deadline"]) <= s:
                    raise ValueError("an amended deadline must lie in the future")
                ob["deadline"] = int(changes["deadline"])
            if "keywords" in changes:
                ob["keywords"] = list(dict.fromkeys(ob["keywords"] + changes["keywords"]))
            ob["status"] = "amended"
            ob["history"].append({"step": s, "event": "amended", "text": body, "before": before})
        else:
            raise ValueError(kind)
        if events is not None:
            events["cancelled" if kind == "cancel" else "amended"].append(ob["id"])
        msg = f"[{ob['id']}] {body}"
        if events is None:
            self.state["outbox"].append(msg)   # delivered with the next observation
        return msg

    def _arrivals(self, u: float, ctx, events) -> list[str]:
        s = ctx.step
        if s < self.cfg.first_request_step or len(self.active()) >= self.cfg.max_concurrent:
            return []
        if u >= self.caretaker.rate(dl.day_index(s)):
            return []
        new = self.caretaker.propose(self.rng, self, ctx)
        if new is None:
            return []
        mark = _pick(self.rng, VALUE_MARK[new["value"]]) if new["family"] != "report" else ""
        ob = self.add_obligation(new["family"], new["spec"], new["deadline"], new["value"], new["text"],
                                 new["keywords"], s, estimate=new["estimate"], progress=new.get("progress"),
                                 mark=mark)
        events["issued"].append(ob["id"])
        return [ob["text"]]

    def add_obligation(self, family: str, spec: dict, deadline: int, value: int, text: str, keywords: list,
                       step: int, estimate: float = 0.0, progress: dict | None = None, mark: str = "") -> dict:
        """Register a new obligation issued at ``step`` (the caretaker's path; public for scripted scenarios).
        The message is ``[id] <mark><text>``; it is evaluated from ``step + 1`` on."""
        from .obligations import FAMILIES
        if family not in FAMILIES:
            raise ValueError(family)
        if int(deadline) <= int(step):
            raise ValueError("deadline must lie after the issue step")
        prefix = "q" if family == "report" else "o"
        oid = f"{prefix}{self.state['next_id']}"
        self.state["next_id"] += 1
        body = text
        if mark:
            first = body.split(" ", 1)[0]
            body = mark + (body if first in ("I", "I'd") else body[0].lower() + body[1:])
        full = f"[{oid}] {body}"
        s = int(step)
        ob = {"id": oid, "family": family, "text": full, "spec": copy.deepcopy(spec), "issue_step": s,
              "deadline": int(deadline), "value": int(value), "status": "open", "resolved_step": None,
              "reason": None, "keywords": list(keywords), "estimate": float(estimate), "day": dl.day_index(s),
              "phase": dl.phase(s), "history": [{"step": s, "event": "issued", "text": full}]}
        if family == "visit":
            ob["progress"] = copy.deepcopy(progress) if progress else {"reached_step": None}
        self.state["obligations"].append(ob)
        self._index[oid] = ob
        return ob

    def _emit(self, step: int, msgs: list[str]) -> list[str]:
        for m in msgs:
            self.state["messages"].append({"step": int(step), "text": m})
        del self.state["messages"][:-self.cfg.log_keep]
        return list(msgs)
