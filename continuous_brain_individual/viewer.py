"""A live window onto the W1 individual (exploration lane, engineering only).

``LiveSession`` runs one life step by step and exposes, after every tick, what the individual sensed,
perceived, remembered, understood, intended, said and did, and how each faculty's state in the canonical store
changed.  ``serve()`` puts it behind a local HTTP server with a single-page UI
(``continuous_brain_individual/viewer_ui/index.html``); ``scripts/explore/w1_viewer.py`` is the command line.

Watching does not change the life.  The viewer only *reads*: the store's channels, the TickResult digests and,
for the human only, W1's privileged ground truth (the true map and creatures), which is labelled as such and
never reaches the individual.  The per-faculty timers wrap each host's ``step`` without touching its inputs or
outputs, and the sharp world render saves and restores Crafter's RNG.  ``tests/test_w1_viewer.py`` checks
that a watched life is identical to an unwatched one.

Operator actions are explicit and logged in the session's operator log:
  * erase a faculty's state (``<slot>.state`` set to empty in the store, effective from the next tick; an
    operator edit made through the runtime's state, not a preregistered intervention);
  * bookmark and return (in memory);
  * save to disk (the runtime through the never-overwrite ``SnapshotStore``, the world as W1's JSON snapshot)
    and resume from such a save.

A lifelong individual (the default here: learned skills, one individual across lives) does not end at death:
its last moment is one more tick in which it knows it is dying (its faculties close the life and consolidate
what they learned), the whole individual is saved to disk automatically, and the next life continues it in a
new world (``next_life``).  Each life's summary is appended to ``<root>/<individual>/lives.jsonl``.
"""
from __future__ import annotations

import base64
import io
import json
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from PIL import Image

from continuous_brain_snn.snapshot import SnapshotStore
from continuous_brain_world import daylight as dl
from continuous_brain_world.env import W1Config, W1Env, action_names
from continuous_brain_organs.language import LanguageOrgan
from continuous_brain_organs.vision import MATERIALS, OBJECTS, labels_from_privileged

from . import codec, ledger, memory as mem
from .body import VITALS, body_sense
from .hosts import OPTION_CODES, SelfModelHost
from .individual import Individual, IndividualConfig, first_info
from .skills import OPTIONS as LEARNED_OPTIONS

UI = Path(__file__).resolve().parent / "viewer_ui" / "index.html"
ACTIONS = action_names()
ROUTINE_OPTIONS = sorted(OPTION_CODES, key=OPTION_CODES.get)
OPTIONS = ROUTINE_OPTIONS
FACULTIES = ("perception", "episodic_memory", "goal_genesis", "self_state", "metacognition", "skills",
             "autobiography")
LEARNED_FACULTIES = FACULTIES[:5] + ("world_model",) + FACULTIES[5:]
FACULTY_NAMES = {"perception": "Perception", "episodic_memory": "Memory", "goal_genesis": "Commitments",
                 "self_state": "Self-model", "metacognition": "Speech", "skills": "Skills",
                 "autobiography": "Autobiography", "driver": "Senses", "world_model": "Foresight"}
DEFAULT_ARM = {"skills": "learned", "lifelong": True}
HISTORY = 4000                      # per-tick records kept for the UI's time lines
LOG = 400                           # conversation and event log entries kept

# The activity units: every small numeric channel a faculty writes each tick, plus three perception measures
# the viewer derives from perception.tiles.  name, faculty, label
UNITS = ([("perception", "material confidence"), ("perception", "creature confidence"),
          ("perception", "view change")]
         + [("episodic_memory", n) for n in ("known fraction", "kinds seen", "objects in view", "steps lived")]
         + [("goal_genesis", f"open {f}") for f in ledger.TASK_FAMILIES] + [("goal_genesis", "open questions")]
         + [("self_state", f"belief {f}") for f in SelfModelHost.FAMILIES]
         + [("self_state", v) for v in VITALS]
         + [("metacognition", "answer confidence")]
         + [("skills", f"option {o}") for o in OPTIONS]
         + [("autobiography", n) for n in ("days narrated", "events today", "commitments closed")]
         + [("autobiography", "expected vs actual")])


def units_for(learned: bool) -> list[tuple[str, str]]:
    if not learned:
        return UNITS
    from .learned import RISKS
    base = [u for u in UNITS if u[0] != "skills"]
    i = next(k for k, u in enumerate(base) if u[0] == "autobiography")
    return (base[:i] + [("world_model", f"risk: {r}") for r in RISKS]
            + [("skills", f"option {o}") for o in LEARNED_OPTIONS] + [("skills", f"value {o}") for o in LEARNED_OPTIONS]
            + base[i:])


def _png(img: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(np.asarray(img, dtype=np.uint8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _grid(arr: np.ndarray, offset: int = 0) -> str:
    """A (W, H) world grid as base64 bytes, row-major by y (so the UI reads [y * W + x])."""
    return base64.b64encode((np.asarray(arr, dtype=np.int16).T + offset).astype(np.uint8).tobytes()).decode()


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_jsonable(v) for v in (sorted(x) if isinstance(x, set) else x)]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    return x


class LiveSession:
    """One life of the individual, steppable, observable, with operator actions."""

    def __init__(self, vision, language_config: dict, snapshot_root: str | Path, world_config: W1Config | None = None):
        self.vision = vision
        self.language_config = dict(language_config)
        self.snapshot_root = Path(snapshot_root)
        self.world_config = world_config or W1Config()
        self.lock = threading.RLock()
        self.ind = self.env = self.obs = self.info = None
        self.bookmark = None

    # ------------------------------------------------------------------------------------------- lives
    def _build(self, arm: dict) -> Individual:
        cfg = IndividualConfig(memory=arm.get("memory", "full"), language=arm.get("language", "organ"),
                               perception=arm.get("perception", "organ"), skills=arm.get("skills", "routine"),
                               lifelong=bool(arm.get("lifelong", False)))
        if cfg.perception == "organ" and self.vision is None:
            raise ValueError("the vision organ's weights are missing; use perception=oracle or install them")
        ind = Individual(cfg, self.vision, LanguageOrgan.from_config(dict(self.language_config)))
        self._ms = {}
        for slot, comp in ind.rt.components.items():                 # timers: inputs and outputs untouched
            def timed(ctx, _step=comp.step, _slot=slot):
                t = time.perf_counter()
                out = _step(ctx)
                self._ms[_slot] = (time.perf_counter() - t) * 1000.0
                return out
            comp.step = timed
        return ind

    @property
    def learned(self) -> bool:
        return self.ind is not None and self.ind.config.skills == "learned"

    @property
    def lifelong(self) -> bool:
        return self.ind is not None and self.ind.config.lifelong

    @property
    def faculties(self) -> tuple[str, ...]:
        return LEARNED_FACULTIES if self.learned else FACULTIES

    @property
    def options(self) -> list[str]:
        return list(LEARNED_OPTIONS) if self.learned else ROUTINE_OPTIONS

    def new_life(self, seed: int, arm: dict | None = None) -> None:
        """A new individual (newborn: nothing learned) in world ``seed``."""
        with self.lock:
            self.arm = dict(DEFAULT_ARM if arm is None else arm)
            self.ind = self._build(self.arm)
            self.env = W1Env(self.world_config)
            self.seed = int(seed)
            self.obs = self.env.reset(self.seed)
            self.info = first_info(self.env)
            self._reset_logs()
            self.lives = []
            self.individual_id = time.strftime("individual-%Y%m%dT%H%M%S")
            self.started = time.time()

    def next_life(self) -> None:
        """The same individual wakes in the next world (lifelong); otherwise a new individual there."""
        with self.lock:
            if not self.lifelong:
                return self.new_life(self.seed + 1, self.arm)
            if not self.done:
                raise ValueError("it is still alive")
            self.ind.next_life()
            self.seed += 1
            self.env = W1Env(self.world_config)
            self.obs = self.env.reset(self.seed)
            self.info = first_info(self.env)
            self.done, self.death, self.prev_tiles = False, None, None
            self.recent_hits = deque(maxlen=60)
            self.bookmark = None
            self.log.append({"step": 0, "who": "world", "kind": "life",
                             "text": f"life {self.ind.life + 1} begins in world {self.seed}"})

    def _reset_logs(self) -> None:
        self.records = deque(maxlen=HISTORY)
        self.log = deque(maxlen=LOG)
        self.operator = []
        self.last = {"action": None, "utterance": None, "events": [], "heard": [], "option": None}
        self.prev_writes = {}
        self.prev_tiles = None
        self.done = False
        self.death = None
        self.recent_hits = deque(maxlen=60)
        self.bookmark = None

    @property
    def alive(self) -> bool:
        return self.ind is not None and not self.done

    # -------------------------------------------------------------------------------------------- step
    def step(self) -> dict | None:
        with self.lock:
            if not self.alive:
                return None
            heard = list(self.obs["messages"])
            step_before = int(self.obs["step"])
            ach_before = dict(self.info["privileged"]["achievements"])
            action, utter, result = self.ind.tick(self.obs, self.info)
            store = self.ind.rt.store
            events = json.loads(np.asarray(store.read("goal_genesis.events"), np.uint8).tobytes().decode() or "[]")
            o_index = int(np.asarray(store.read("skills.selected_option"))[0])
            option = self.options[o_index]
            self.obs, reward, self.done, self.info = self.env.step(action, utter)
            changed = sorted({n.split(".")[0] for n, h in result.writes.items()
                              if not n.startswith("driver.") and self.prev_writes.get(n) != h})
            self.prev_writes = dict(result.writes)
            units = self._units(store)
            ev = self.info["events"]
            hits = {"zombie": int(ev.get("zombie_hits") or 0), "arrow": int(ev.get("arrow_hits") or 0)}
            self.recent_hits.append(hits)
            inv = self.info["privileged"]["inventory"]
            rec = {"t": int(result.tick), "s": step_before, "a": action, "o": o_index,
                   "v": [int(inv[k]) for k in VITALS], "ms": {k: round(v, 2) for k, v in self._ms.items()},
                   "ch": changed, "u": units, "r": round(float(reward["total"]), 3),
                   "hit": hits["zombie"] + hits["arrow"], "heard": len(heard), "said": bool(utter)}
            self.records.append(rec)
            unlocked = sorted(k for k, v in self.info["privileged"]["achievements"].items()
                              if v and not ach_before.get(k))
            rec["ach"] = unlocked
            self._log_step(step_before, heard, events, utter, ev, unlocked)
            self.last = {"action": ACTIONS[action], "utterance": utter, "events": events, "heard": heard,
                         "option": option}
            if self.done:
                self.death = self._death(step_before)
                self.log.append({"step": step_before, "who": "world", "kind": "death", "text": self.death["text"]})
                if self.lifelong:
                    self._last_moment()
            return rec

    def _last_moment(self) -> None:
        """Lifelong: its dying tick (it closes the life and consolidates), then the life's record and a save."""
        result = self.ind.die(self.obs, self.info)
        store = self.ind.rt.store
        changed = sorted({n.split(".")[0] for n, h in result.writes.items()
                          if not n.startswith("driver.") and self.prev_writes.get(n) != h})
        self.prev_writes = dict(result.writes)
        inv = self.info["privileged"]["inventory"]
        self.records.append({"t": int(result.tick), "s": int(self.obs["step"]), "a": 0,
                             "o": int(np.asarray(store.read("skills.selected_option"))[0]),
                             "v": [int(inv[k]) for k in VITALS], "ms": {k: round(v, 2) for k, v in self._ms.items()},
                             "ch": changed, "u": self._units(store), "r": 0.0, "hit": 0, "heard": 0, "said": False,
                             "ach": [], "death": True})
        summary = self.env.episode_summary()
        rec = {"life": self.ind.life, "seed": self.seed, "steps": int(self.obs["step"]),
               "done": summary["crafter"]["n_unlocked"], "fulfilled": summary["obligations"]["fulfilled"],
               "failed": summary["obligations"]["failed"], "memory": [summary["memory"]["correct"],
                                                                      summary["memory"]["asked"]],
               "cause": self.death["text"]}
        if self.learned:
            ctl = self.ind.read_state("skills")["ctl"]
            rec.update(options=list(ctl["life_options"]), life_return=ctl["returns"][-1] if ctl["returns"] else None,
                       decisions=ctl["decisions"], updates=ctl["updates"])
        self.lives.append(rec)
        root = self.snapshot_root / self.individual_id
        root.mkdir(parents=True, exist_ok=True)
        with open(root / "lives.jsonl", "a") as f:
            f.write(json.dumps(_jsonable(rec)) + "\n")
        try:
            snap = self.save()
            self.log.append({"step": rec["steps"], "who": "operator", "kind": "save",
                             "text": f"saved automatically after life {self.ind.life + 1}: {snap.name}"})
        except Exception as e:                                          # never lose the life over a save error
            self.log.append({"step": rec["steps"], "who": "operator", "kind": "save",
                             "text": f"automatic save failed: {type(e).__name__}: {e}"})

    def _units(self, store) -> list[float]:
        tiles = codec.unblob(store.read("perception.tiles"))
        m, o = np.asarray(tiles["material"]), np.asarray(tiles["object"])
        change = 0.0 if self.prev_tiles is None else float(((m != self.prev_tiles[0]) | (o != self.prev_tiles[1])).mean())
        self.prev_tiles = (m, o)
        out = [float(np.mean(tiles["material_conf"])), float(np.mean(tiles["object_conf"])), change]
        for ch in ("episodic_memory.slots", "goal_genesis.active_goal", "self_state.vector", "metacognition.confidence"):
            out += [float(x) for x in np.asarray(store.read(ch)).ravel()]
        opt = int(np.asarray(store.read("skills.selected_option"))[0])
        if self.learned:
            out += [float(x) for x in np.asarray(store.read("world_model.prediction")).ravel()]
        out += [1.0 if i == opt else 0.0 for i in range(len(self.options))]
        if self.learned:
            out += [float(x) for x in np.asarray(store.read("skills.values")).ravel()]
        out += [float(x) for x in np.asarray(store.read("autobiography.record")).ravel()]
        out += [float(np.asarray(store.read("autobiography.discrepancy"))[0])]
        return [round(x, 4) for x in out]

    def _log_step(self, step, heard, events, utter, ev, unlocked) -> None:
        by_msg = {}
        for e in events:
            by_msg.setdefault(e.get("message"), []).append({k: v for k, v in e.items() if k != "provenance"})
        for m in heard:
            self.log.append({"step": step, "who": "caretaker", "kind": "message", "text": m,
                             "understood": by_msg.get(m, [])})
        if utter:
            self.log.append({"step": step, "who": "individual", "kind": "speech", "text": utter})
        for a in ev.get("answers", []) or []:
            if a.get("event") == "answered":
                self.log.append({"step": step, "who": "world", "kind": "verdict",
                                 "text": f"answer to {a['id']} was {'right' if a.get('correct') else 'wrong'}",
                                 "ok": bool(a.get("correct"))})
        for key, ok in (("fulfilled", True), ("failed", False)):
            for oid in ev.get(key, []) or []:
                if any(a.get("id") == oid for a in ev.get("answers", []) or []):
                    continue
                self.log.append({"step": step, "who": "world", "kind": "verdict",
                                 "text": f"{oid} {'fulfilled' if ok else 'failed'}", "ok": ok})
        for k in unlocked:
            self.log.append({"step": step, "who": "world", "kind": "achievement",
                             "text": f"first time: {k.replace('_', ' ')}"})

    def _death(self, step) -> dict:
        inv = self.info["privileged"]["inventory"]
        starving = [k for k in ("food", "drink", "energy") if inv.get(k, 1) == 0]
        z = sum(h["zombie"] for h in list(self.recent_hits)[-30:])
        a = sum(h["arrow"] for h in list(self.recent_hits)[-30:])
        parts = []
        if starving:
            parts.append("out of " + ", ".join({"food": "food", "drink": "water", "energy": "sleep"}[k] for k in starving))
        if z:
            parts.append(f"{z} zombie hit{'s' * (z > 1)} in its last 30 steps")
        if a:
            parts.append(f"{a} arrow hit{'s' * (a > 1)} in its last 30 steps")
        text = f"died at step {step}" + (": " + "; ".join(parts) if parts else "")
        return {"step": step, "text": text, "starving": starving, "zombie_hits": z, "arrow_hits": a}

    # ------------------------------------------------------------------------------------------ views
    def meta(self) -> dict:
        with self.lock:
            graph = {"nodes": [{"id": "driver", "name": "Senses"}] +
                              [{"id": s, "name": FACULTY_NAMES[s]} for s in self.faculties] + [{"id": "world", "name": "World"}],
                     "edges": []}
            if self.ind is not None:
                owners = self.ind.rt.store.owners()
                seen = set()
                for slot, comp in self.ind.rt.components.items():
                    for ch in tuple(comp.reads) + tuple(getattr(comp, "reads_delayed", ())):
                        src = owners[ch]
                        if src == slot:
                            continue
                        key = (src, slot, ch in getattr(comp, "reads_delayed", ()))
                        if key not in seen:
                            seen.add(key)
                            graph["edges"].append({"from": src, "to": slot, "delayed": key[2], "channel": ch})
                graph["edges"].append({"from": "skills", "to": "world", "delayed": False, "channel": "skills.action"})
                graph["edges"].append({"from": "metacognition", "to": "world", "delayed": False,
                                       "channel": "metacognition.utterance"})
                graph["edges"].append({"from": "world", "to": "driver", "delayed": False, "channel": "W1 step"})
                decls = {n: self.ind.rt.store.decl(n).description for n in self.ind.rt.store.names()}
                lang = self.ind.language.provenance() if hasattr(self.ind.language, "provenance") else {}
            else:
                decls, lang = {}, {}
            risks = []
            if self.learned:
                from .learned import RISKS
                risks = list(RISKS)
            return {"format": "cb-w1-viewer/1", "status": "ENGINEERING_ONLY", "actions": ACTIONS, "options": self.options,
                    "learned": self.learned, "lifelong": self.lifelong, "risks": risks,
                    "materials": list(MATERIALS), "objects": list(OBJECTS), "vitals": list(VITALS),
                    "faculties": list(self.faculties), "faculty_names": FACULTY_NAMES,
                    "units": [{"faculty": f, "label": l} for f, l in units_for(self.learned)], "graph": graph,
                    "channels": decls,
                    "language": _jsonable(lang), "vision_weights": getattr(self.vision, "weights_sha256", None),
                    "families": list(SelfModelHost.FAMILIES), "day_length": dl.DAY_LENGTH,
                    "daylight_curve": [round(dl.daylight(t), 3) for t in range(0, dl.DAY_LENGTH, 5)],
                    "view": {"rows": mem.ROWS, "cols": mem.COLS, "player": [mem.PR, mem.PC]},
                    "world_size": [mem.W, mem.H]}

    def records_since(self, tick: int) -> list[dict]:
        with self.lock:
            return [r for r in self.records if r["t"] > tick]

    def frame(self) -> dict:
        with self.lock:
            if self.ind is None:
                return {"alive": False, "tick": -1}
            rt, priv, step = self.ind.rt, self.info["privileged"], int(self.obs["step"])
            memory = self.ind.read_state("episodic_memory") or mem.empty()
            led = self.ind.read_state("goal_genesis") or ledger.empty()
            skills = self.ind.read_state("skills") or {}
            selfm = self.ind.read_state("self_state") or {}
            speech = self.ind.read_state("metacognition") or {}
            auto = self.ind.read_state("autobiography") or {}
            # What it perceives in the frame on screen: the organ is a fixed function of the pixels, so this is
            # exactly what its perception faculty will write at the next tick (perception.tiles in the store
            # holds the previous frame's, one world step behind the picture).
            true_mat, true_obj = labels_from_privileged(priv)
            if self.ind.config.perception == "oracle":
                tiles = {"material": true_mat, "object": true_obj, "material_conf": np.ones(true_mat.shape),
                         "object_conf": np.ones(true_obj.shape)}
            else:
                tiles = self.vision.perceive(np.asarray(self.obs["image"], dtype=np.uint8))
            body = body_sense(self.info)
            perceived = None
            if tiles is not None:
                pm, po = np.asarray(tiles["material"]), np.asarray(tiles["object"])
                perceived = {"material": pm.tolist(), "object": po.tolist(),
                             "material_conf": np.round(np.asarray(tiles["material_conf"]), 3).tolist(),
                             "object_conf": np.round(np.asarray(tiles["object_conf"]), 3).tolist(),
                             "accuracy": {"material": float((pm == true_mat).mean()),
                                          "object": float((po == true_obj).mean())}}
            store = []
            for name in rt.store.names():
                v = np.asarray(rt.store.read(name))
                store.append({"name": name, "owner": rt.store.owners()[name], "bytes": int(v.nbytes),
                              "digest": (self.prev_writes.get(name) or "")[:10]})
            commitments = sorted(led["commitments"].values(), key=lambda c: (c["status"] != "open", -c["issue_step"]))
            truth_ob = {o["id"]: o for o in priv["obligations"]}
            truth_hist = {o["id"]: o["status"] for o in priv["history"]}
            return _jsonable({
                "alive": not self.done, "death": self.death, "tick": rt.tick, "step": step, "seed": self.seed,
                "arm": self.ind.config.arm(), "day": dl.day_index(step), "phase": dl.phase(step),
                "time": dl.describe(step), "daylight": round(float(priv["daylight"]), 3),
                "dark": bool(dl.is_dark(step)),
                "image": _png(self.obs["image"]),
                "body": {"vitals": {k: int(body["inventory"][k]) for k in VITALS},
                         "inventory": {k: int(v) for k, v in body["inventory"].items() if k not in VITALS and v},
                         "pos": body["player_pos"], "facing": body["facing"], "sleeping": body["sleeping"],
                         "achievements": sorted(k for k, v in body["achievements"].items() if v)},
                "perceived": perceived, "truth_view": {"material": true_mat.tolist(), "object": true_obj.tolist()},
                "memory": {"map": _grid(memory["map"], 1), "objects": mem.believed_objects(memory),
                           "last_seen": memory["last_seen"], "known": float((memory["map"] >= 0).mean()),
                           "blocked": len(memory.get("blocked", {}))},
                "world": {"map": _grid(priv["material_map"]),
                          "objects": [{"kind": o["kind"], "pos": list(o["pos"])} for o in priv["objects"]
                                      if o["kind"] != "player"]},
                "ledger": {"commitments": commitments, "questions": list(led["questions"].values()),
                           "unparsed": led["unparsed"][-5:], "organ_calls": led["organ_calls"],
                           "active": [c["id"] for c in ledger.active(led, step)]},
                "caretaker": {oid: {"status": o["status"], "progress": o.get("progress"), "deadline": o["deadline"]}
                              for oid, o in truth_ob.items()} | {oid: {"status": s} for oid, s in truth_hist.items()
                                                                  if oid not in truth_ob},
                "skills": {"action": self.last["action"], "option": self.last["option"],
                           "task": skills.get("last_task"), "drinking": skills.get("drinking"),
                           "eating": skills.get("eating"), "shelter": skills.get("shelter"),
                           "explore_target": skills.get("explore_target"), "bad_spots": skills.get("bad_spots", []),
                           "caution_until": skills.get("caution_until", -1),
                           "visit_reached": skills.get("visit_reached", {})},
                "self_model": selfm, "speech": {"answered": len(speech.get("answered", [])),
                                                "unanswerable": speech.get("unanswerable", [])[-5:],
                                                "last": self.last["utterance"]},
                "autobiography": {"days": auto.get("days", []), "today": len(auto.get("today", [])),
                                  "expected": auto.get("expected", 0.0), "actual": auto.get("actual", 0),
                                  "closed": auto.get("closed", 0)},
                "learning": self._learning() if self.learned else None,
                "life": self.ind.life, "lifelong": self.lifelong, "lives": self._lives(auto)[-300:],
                "individual": getattr(self, "individual_id", None),
                "store": store, "log": list(self.log)[-120:], "operator": self.operator[-20:],
                "bookmark": None if self.bookmark is None else self.bookmark["tick"],
                "state_hash": None,
            })

    def _lives(self, auto: dict) -> list[dict]:
        """The lives lived in this viewer; for an adopted individual, its own record of the lives before."""
        own = [{"life": l["life"], "steps": l["steps"], "done": l["done"], "fulfilled": None, "failed": None,
                "cause": "died: " + (", ".join(l["felt"]) if l.get("felt") else "hurt")}
               for l in auto.get("lives", []) if l["life"] < (self.lives[0]["life"] if self.lives else self.ind.life)]
        return own + list(self.lives)

    def _learning(self) -> dict:
        from .learned import RISKS, LearnedSkillHost
        store = self.ind.rt.store
        ctl = (self.ind.read_state("skills") or {}).get("ctl") or {}
        wm = self.ind.read_state("world_model") or {}
        host = self.ind.rt.components["skills"]
        return {"risks": list(RISKS), "prediction": np.round(np.asarray(store.read("world_model.prediction")), 4),
                "options": list(LEARNED_OPTIONS), "values": np.round(np.asarray(store.read("skills.values")), 3),
                "selected": int(np.asarray(store.read("skills.selected_option"))[0]),
                "decisions": ctl.get("decisions", 0), "updates": ctl.get("updates", 0), "loss": ctl.get("loss"),
                "advantage": np.round(np.asarray(store.read("skills.advantage")), 3),
                "exploration": LearnedSkillHost.exploration(host, ctl) if ctl else None,
                "learning_on": bool(host.learn), "lives_lived": ctl.get("lives", 0),
                "replay": ctl.get("count", 0), "returns": ctl.get("returns", [])[-300:],
                "foresight": {"samples": wm.get("samples", 0), "loss": wm.get("loss"), "brier": wm.get("brier"),
                              "memory": wm.get("count", 0), "updates": wm.get("updates", 0)}}

    def view_jpeg(self, size: int = 384) -> bytes:
        with self.lock:
            if self.env is None:
                return b""
            img = self.env.bridge.render(int(size))            # RNG saved and restored by the bridge
        buf = io.BytesIO()
        Image.fromarray(np.asarray(img, dtype=np.uint8)).save(buf, format="JPEG", quality=88)
        return buf.getvalue()

    # --------------------------------------------------------------------------------- operator actions
    def erase(self, slot: str) -> str:
        if slot not in self.faculties or slot == "perception":
            raise ValueError(f"cannot erase {slot!r}")
        with self.lock:
            st = self.ind.rt.state_dict()
            st["store"]["current"][f"{slot}.state"] = np.zeros(0, np.uint8)
            self.ind.rt.load_state_dict(st)
            note = f"tick {self.ind.rt.tick}: erased {slot}.state (the {FACULTY_NAMES[slot].lower()} starts from nothing next tick)"
            self.operator.append(note)
            self.log.append({"step": int(self.obs["step"]), "who": "operator", "kind": "erase", "text": note})
            return note

    def mark(self) -> str:
        with self.lock:
            self.bookmark = {"tick": self.ind.rt.tick, "runtime": self.ind.save(), "world": self.env.snapshot(),
                             "obs": dict(self.obs), "info": self.info, "done": self.done, "death": self.death,
                             "records": list(self.records), "log": list(self.log),
                             "prev_writes": dict(self.prev_writes), "prev_tiles": self.prev_tiles,
                             "hits": list(self.recent_hits)}
            note = f"tick {self.ind.rt.tick}: bookmarked"
            self.operator.append(note)
            return note

    def back(self) -> str:
        with self.lock:
            b = self.bookmark
            if b is None:
                raise ValueError("no bookmark")
            import copy
            self.ind.restore(copy.deepcopy(b["runtime"]))
            self.env.restore(copy.deepcopy(b["world"]))
            self.obs, self.info, self.done, self.death = dict(b["obs"]), b["info"], b["done"], b["death"]
            self.records = deque(b["records"], maxlen=HISTORY)
            self.log = deque(b["log"], maxlen=LOG)
            self.prev_writes, self.prev_tiles = dict(b["prev_writes"]), b["prev_tiles"]
            self.recent_hits = deque(b["hits"], maxlen=60)
            note = f"returned to the bookmark at tick {b['tick']}"
            self.operator.append(note)
            return note

    def save(self, name: str | None = None) -> Path:
        """Save the life to disk: <root>/<name>/checkpoints/<snapshot>/ (never overwritten) plus world and
        viewer files beside the snapshot directory.  Returns the snapshot directory."""
        with self.lock:
            name = name or getattr(self, "individual_id", None) or f"seed{self.seed}"
            store = SnapshotStore(self.snapshot_root / name)
            snap = self.ind.rt.save(store, label=f"tick{self.ind.rt.tick}")
            side = {"format": "cb-w1-viewer-life/1", "status": "ENGINEERING_ONLY", "seed": self.seed, "arm": self.arm,
                    "language": self.language_config, "tick": self.ind.rt.tick,
                    "state_hash": self.ind.state_hash(),
                    "obs": {"image": base64.b64encode(np.asarray(self.obs["image"], np.uint8).tobytes()).decode(),
                            "shape": list(np.asarray(self.obs["image"]).shape),
                            **{k: self.obs[k] for k in ("messages", "step", "day", "phase")}},
                    "done": self.done, "death": self.death, "log": list(self.log), "life": self.ind.life,
                    "lives": self.lives, "individual_id": getattr(self, "individual_id", None)}
            world = Path(str(snap) + ".world.json")
            viewer = Path(str(snap) + ".viewer.json")
            for p, text in ((world, W1Env.snapshot_to_json(self.env.snapshot())), (viewer, json.dumps(_jsonable(side)))):
                if p.exists():
                    raise FileExistsError(p)
                p.write_text(text)
            self.operator.append(f"tick {self.ind.rt.tick}: saved to {snap}")
            return snap

    def resume(self, snapshot: str | Path) -> None:
        snapshot = Path(snapshot)
        side = json.loads(Path(str(snapshot) + ".viewer.json").read_text())
        with self.lock:
            self.arm = dict(side["arm"])
            self.ind = self._build(self.arm)
            self.ind.rt.restore(SnapshotStore(snapshot.parent.parent), snapshot)
            if self.ind.state_hash() != side["state_hash"]:
                raise ValueError("restored state hash differs from the saved one")
            self.env = W1Env(self.world_config)
            self.env.restore(W1Env.snapshot_from_json(Path(str(snapshot) + ".world.json").read_text()))
            o = side["obs"]
            self.obs = {"image": np.frombuffer(base64.b64decode(o["image"]), np.uint8).reshape(o["shape"]).copy(),
                        **{k: o[k] for k in ("messages", "step", "day", "phase")}}
            self.info = self.env.privileged_now()
            self.seed = int(side["seed"])
            self._reset_logs()
            self.ind.life = int(side.get("life", 0))
            self.lives = list(side.get("lives", []))
            self.individual_id = side.get("individual_id") or snapshot.parent.parent.name
            self.log.extend(side.get("log", []))
            self.done, self.death = bool(side["done"]), side["death"]
            self.operator.append(f"resumed from {snapshot} at tick {self.ind.rt.tick}")
            self.started = time.time()

    def adopt(self, snapshot: str | Path, seed: int) -> None:
        """Watch an individual trained by scripts/explore/w1_lifelong.py: its saved runtime (everything it has
        learned and believes) wakes in world ``seed`` as its next life."""
        snapshot = Path(snapshot)
        meta = json.loads(Path(str(snapshot) + ".meta.json").read_text())
        with self.lock:
            self.arm = dict(DEFAULT_ARM)
            self.ind = self._build(self.arm)
            self.ind.rt.restore(SnapshotStore(snapshot.parent.parent), snapshot)
            if self.ind.state_hash() != meta["state_hash"]:
                raise ValueError("restored state hash differs from the saved one")
            self.ind.life = int(meta["life"])
            self.env = W1Env(self.world_config)
            self.seed = int(seed)
            self.obs = self.env.reset(self.seed)
            self.info = first_info(self.env)
            self._reset_logs()
            self.lives = []
            self.individual_id = f"adopted-{snapshot.name}"
            self.log.append({"step": 0, "who": "operator", "kind": "adopt",
                             "text": f"adopted after {meta['lives_lived']} lives of training ({snapshot.name})"})
            self.started = time.time()


# --------------------------------------------------------------------------------------------- the server
class Runner(threading.Thread):
    """Steps the session in the background at the requested pace."""

    def __init__(self, session: LiveSession):
        super().__init__(daemon=True)
        self.session = session
        self.playing = False
        self.speed = 10.0               # steps per second; 0 = as fast as it can
        self.auto_next = False
        self.stop_flag = False
        self.error = None

    def run(self):
        while not self.stop_flag:
            if not self.playing or not self.session.alive:
                if self.playing and self.auto_next and self.session.ind is not None and self.session.done:
                    self.session.next_life()
                    continue
                if self.playing and not self.session.alive:
                    self.playing = False
                time.sleep(0.05)
                continue
            t = time.perf_counter()
            try:
                self.session.step()
            except Exception as e:                       # surface in the UI, stop playing
                self.error = f"{type(e).__name__}: {e}"
                self.playing = False
                continue
            if self.speed > 0:
                time.sleep(max(0.0, 1.0 / self.speed - (time.perf_counter() - t)))


def make_handler(session: LiveSession, runner: Runner):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body: bytes, ctype="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj).encode())

        def do_GET(self):
            path, _, query = self.path.partition("?")
            q = dict(p.split("=", 1) for p in query.split("&") if "=" in p)
            if path in ("/", "/index.html"):
                return self._send(200, UI.read_bytes(), "text/html; charset=utf-8")
            if path == "/api/meta":
                return self._json(session.meta())
            if path == "/api/frame":
                f = session.frame()
                f["records"] = session.records_since(int(q.get("since", -1)))
                f["runner"] = {"playing": runner.playing, "speed": runner.speed, "auto_next": runner.auto_next,
                               "error": runner.error}
                return self._json(f)
            if path == "/api/view.jpg":
                return self._send(200, session.view_jpeg(int(q.get("size", 384))), "image/jpeg")
            return self._send(404, b"not found", "text/plain")

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            try:
                cmd = json.loads(self.rfile.read(n) or b"{}")
                c = cmd.get("cmd")
                out = {"ok": True}
                if c == "play":
                    runner.playing, runner.error = True, None
                elif c == "pause":
                    runner.playing = False
                elif c == "step":
                    runner.playing = False
                    for _ in range(int(cmd.get("n", 1))):
                        session.step()
                elif c == "speed":
                    runner.speed = float(cmd["value"])
                elif c == "auto_next":
                    runner.auto_next = bool(cmd["value"])
                elif c == "new_life":
                    runner.playing = False
                    session.new_life(int(cmd.get("seed", session.seed + 1)), cmd.get("arm"))
                    out["meta"] = session.meta()
                elif c == "next_life":
                    session.next_life()
                    out["meta"] = session.meta()
                elif c == "erase":
                    out["note"] = session.erase(cmd["slot"])
                elif c == "bookmark":
                    out["note"] = session.mark()
                elif c == "back":
                    runner.playing = False
                    out["note"] = session.back()
                elif c == "save":
                    out["note"] = f"saved to {session.save(cmd.get('name'))}"
                else:
                    raise ValueError(f"unknown command {c!r}")
                return self._json(out)
            except Exception as e:
                return self._json({"ok": False, "error": f"{type(e).__name__}: {e}"}, 400)

    return Handler


def latest_snapshot(root: str | Path) -> Path | None:
    """The most recently written snapshot (with its viewer file) under ``root``."""
    snaps = [Path(str(p)[:-len(".viewer.json")]) for p in Path(root).glob("*/checkpoints/*.viewer.json")]
    return max(snaps, key=lambda p: Path(str(p) + ".viewer.json").stat().st_mtime) if snaps else None


def serve(session: LiveSession, host: str = "127.0.0.1", port: int = 8765, play: bool = False):
    runner = Runner(session)
    runner.playing = play
    runner.start()
    httpd = ThreadingHTTPServer((host, port), make_handler(session, runner))
    return httpd, runner
