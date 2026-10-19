"""Evaluation harness of the language organ: a labelled corpus from W1's own caretaker, and its scoring.

EVALUATION ONLY.  This module reads W1's privileged layer state (the obligation records, their history, the
visibility log) to build ground truth; the individual must never import it.

Corpus: W1 episodes are run with W1's privileged scripted baseline as the actor (it keeps the agent alive long
enough for days, nights, amendments, cancellations and every question type to happen; its actions do not
affect what the caretaker *says* except through the world state).  By default its answers are made noisy
(``NoisyAnswers``) so that the "wrong answer" and "no answer" feedback frames occur too.  Every caretaker message of every step
becomes one item::

    {"seed", "step", "message",
     "context": {"step", "inventory", "achievements", "player_pos"},   # what the individual itself can know
     "truth": <ground-truth event, organ schema form>,                   # from the layer's records
     "qa": {...}}                                                        # questions only: oracle memory + truth

``context`` holds the state the individual perceives or tracks itself (the inventory strip; its own counters
and position); the organ fills W1's state-derived fields from it.  ``commitments`` (the organ's own earlier
outputs) is added by the scorer while it replays the corpus in order, per seed.

``qa.memory`` is an **oracle memory** built from the layer (what a perfect memory would hold).  Answers composed
from it measure the answer composer and W1's grammar round trip, not the individual's memory.
"""
from __future__ import annotations

import copy
import statistics
import time

from continuous_brain_world import W1Config, W1Env
from continuous_brain_world import daylight as dl
from continuous_brain_world.caretaker import COUNT_EVENTS
from continuous_brain_world.obligations import chebyshev, grade_report, report_truth
from continuous_brain_world.utterance import parse

from .language import ANSWER_TYPES, OrganError, message_tag, resolve_deadline

SEED_FAMILY = "w1-language-eng"
SEED_BASE = 72000


# --- the actor ----------------------------------------------------------------------------------------
class NoisyAnswers:
    """W1's privileged scripted baseline, except that each of its report answers is, with seeded probability,
    withheld (the caretaker later says "You did not answer in time") or replaced by a wrong number (the
    caretaker says "Not quite. The answer was: ...").  Only so the corpus contains every feedback frame."""

    def __init__(self, seed: int, p_withhold: float = 0.15, p_wrong: float = 0.15):
        import numpy as np
        from continuous_brain_world.baselines import ScriptedAgent
        self.inner = ScriptedAgent(seed=seed)
        self.rng = np.random.default_rng([int(seed), 0x1A9])
        self.p_withhold, self.p_wrong = p_withhold, p_wrong

    def reset(self, obs=None, info=None):
        self.inner.reset(obs, info)

    def act(self, obs, info):
        action, utter = self.inner.act(obs, info)
        if not utter:
            return action, utter
        out = []
        for clause in utter.split("; "):
            if clause.startswith("answer "):
                u = float(self.rng.random())
                if u < self.p_withhold:
                    continue
                if u < self.p_withhold + self.p_wrong:
                    clause = clause.split(":", 1)[0] + ": 17"
            out.append(clause)
        return action, "; ".join(out) or None


# --- labelling -----------------------------------------------------------------------------------------
def _spec_truth(ob: dict) -> dict:
    fam, s = ob["family"], ob["spec"]
    if fam == "collect":
        return {"item": s["item"], "quantity": s["quantity"], "base": s["base"], "target": s["target"]}
    if fam == "craft":
        return {"tool": s["tool"], "base": s["base"]}
    if fam == "place":
        out = {"what": s["what"], "base": s["base"]}
        if "near" in s:
            out.update(near=s["near"], radius=s["radius"])
        else:
            out["onto"] = s["onto"]
        return out
    if fam == "maintain":
        return {"stat": s["stat"], "threshold": s["threshold"]}
    if fam == "avoid":
        return {"source": s["source"]}
    # visit: the caretaker states a rounded distance only in the "about N tiles away" frame
    dist = chebyshev(s["example_target"], s["start"])
    stated = int(round(dist / 5.0) * 5) or dist
    return {"feature": s["feature"], "direction": s["direction"],
            "stated_distance": stated if "tiles away" in ob["text"] else None, "start": list(s["start"])}


_AMEND_FIELDS = ("quantity", "target", "tool", "base", "threshold")


def label_message(env: W1Env, message: str, step: int, priv: dict) -> dict:
    """Ground truth of one caretaker message heard at ``step`` (call right after the step that produced it)."""
    tag = message_tag(message)
    if tag is None:
        return {"type": "info"}
    ob = env.layer._index[tag]
    body = message[len(f"[{tag}] "):]
    if ob["issue_step"] == step and message == ob["text"]:
        if ob["family"] == "report":
            return {"type": "question", "id": tag, "spec": copy.deepcopy(ob["spec"]), "deadline_step": ob["deadline"],
                    "answer_type": ANSWER_TYPES[ob["spec"]["question"]]}
        return {"type": "new_commitment", "id": tag, "family": ob["family"], "value": ob["value"],
                "deadline_step": ob["deadline"], "spec": _spec_truth(ob)}
    for h in ob["history"]:
        if h["step"] != step or h.get("text") != body:
            continue
        if h["event"] == "cancelled":
            return {"type": "cancellation", "id": tag, "family": ob["family"],
                    "reference": body[len("Never mind the "):-1] if body.startswith("Never mind the ") else
                    (body[len("Forget about the "):].split(",")[0] if body.startswith("Forget about the ") else None)}
        if h["event"] == "amended":
            before, after = h["before"], {**{k: ob["spec"].get(k) for k in _AMEND_FIELDS if k in ob["spec"]},
                                          "deadline_step": ob["deadline"]}
            diff = [k for k in _AMEND_FIELDS if k in ob["spec"] and ob["spec"][k] != before["spec"].get(k)]
            if ob["deadline"] != before["deadline"]:
                diff.append("deadline_step")
            return {"type": "amendment", "id": tag, "family": ob["family"], "after": after, "diff": diff}
    if ob["resolved_step"] == step:
        rep = ob["family"] == "report"
        reason = ob["reason"]
        outcome = {"target_reached": "fulfilled", "made": "fulfilled", "placed": "fulfilled", "kept": "fulfilled",
                   "returned": "fulfilled", "deadline": "late", "violated": "violated", "correct": "correct",
                   "wrong": "wrong", "no_answer": "no_answer"}[reason]
        revealed = None
        if rep and reason == "wrong" and "The answer was:" in message:
            revealed = ob["answer"]["truth"]["text"]
        elif rep and reason == "no_answer" and "The answer was:" in message:
            revealed = report_truth(ob, env.layer.state, priv["player_pos"], priv["achievements"], step)["text"]
        mentioned = ob["resolved_step"] if reason == "violated" else (ob["deadline"] if reason == "deadline" else None)
        return {"type": "feedback", "id": tag, "family": ob["family"], "outcome": outcome,
                "step_mentioned": mentioned, "revealed_answer": revealed}
    raise RuntimeError(f"cannot label message {message!r} at step {step}")


def oracle_memory(env: W1Env, priv: dict, step: int) -> dict:
    st = env.layer.state
    day = dl.day_index(step)
    snap = st["day_achievements"][str(day)]
    return {"position": list(priv["player_pos"]),
            "last_seen": copy.deepcopy(st["visibility"]["last_seen"]),
            "counts_today": {e: int(priv["achievements"][e] - snap[e]) for e in COUNT_EVENTS},
            "requests": [{"id": o["id"], "issue_step": o["issue_step"]} for o in st["obligations"]
                         if o["family"] != "report"],
            "cancellations": [{"id": o["id"], "step": o["resolved_step"]} for o in st["obligations"]
                              if o["status"] == "cancelled"]}


def build_corpus(seeds, episode_length: int = 3000, config: dict | None = None, max_messages: int | None = None,
                 actor: str = "noisy_scripted") -> dict:
    """Run W1 on ``seeds`` and label every caretaker message.  ``config`` overrides W1Config fields."""
    from continuous_brain_world.baselines import RandomAgent, ScriptedAgent
    cfg = W1Config(episode_length=episode_length, **(config or {}))
    items, t0 = [], time.perf_counter()
    for seed in seeds:
        env = W1Env(cfg)
        obs = env.reset(seed)
        agent = {"scripted": lambda: ScriptedAgent(seed=seed), "noisy_scripted": lambda: NoisyAnswers(seed),
                 "random": lambda: RandomAgent(seed=seed)}[actor]()
        agent.reset(obs)
        info = env.privileged_now()
        done = False
        while True:
            priv = info["privileged"]
            s = obs["step"]
            for msg in obs["messages"]:
                truth = label_message(env, msg, s, priv)
                item = {"seed": int(seed), "step": int(s), "message": msg, "truth": truth,
                        "context": {"step": int(s), "inventory": dict(priv["inventory"]),
                                    "achievements": dict(priv["achievements"]),
                                    "player_pos": [int(priv["player_pos"][0]), int(priv["player_pos"][1])]}}
                if truth["type"] == "question":
                    ob = env.layer._index[truth["id"]]
                    item["qa"] = {"memory": oracle_memory(env, priv, s),
                                  "truth": report_truth(ob, env.layer.state, priv["player_pos"], priv["achievements"], s),
                                  "player_pos": list(priv["player_pos"]),
                                  "tolerance": cfg.location_tolerance}
                items.append(item)
                if max_messages and len(items) >= max_messages:
                    done = True
                    break
            if done:
                break
            action, utter = agent.act(obs, info)
            obs, _, ended, info = env.step(action, utter)
            if ended:
                # the final observation's messages are labelled too
                priv = info["privileged"]
                for msg in obs["messages"]:
                    if max_messages and len(items) >= max_messages:
                        break
                    items.append({"seed": int(seed), "step": int(obs["step"]), "message": msg,
                                  "truth": label_message(env, msg, obs["step"], priv),
                                  "context": {"step": int(obs["step"]), "inventory": dict(priv["inventory"]),
                                              "achievements": dict(priv["achievements"]),
                                              "player_pos": [int(priv["player_pos"][0]), int(priv["player_pos"][1])]}})
                break
        if done:
            break
    return {"format": "cb-language-corpus/1", "seed_family": SEED_FAMILY, "seeds": [int(s) for s in seeds],
            "episode_length": episode_length, "config_overrides": config or {}, "actor": actor,
            "w1_config": cfg.to_dict(), "build_seconds": time.perf_counter() - t0, "items": items}


# --- scoring -------------------------------------------------------------------------------------------
def group_of(truth: dict) -> str:
    t = truth["type"]
    if t == "new_commitment":
        return truth["family"]
    if t == "question":
        return truth["spec"]["question"]
    if t in ("amendment", "cancellation", "feedback"):
        return truth["family"]
    return t


def score_item(item: dict, events: list) -> dict:
    """Per-field comparison of the organ's events for one message with the ground truth."""
    truth, msg = item["truth"], item["message"]
    fields: dict = {}
    pred = events[0] if events else {"type": "none"}
    fields["type"] = pred["type"] == truth["type"]
    t = truth["type"]
    if t != "info":
        fields["id"] = pred.get("id") == truth["id"]
    same = fields["type"]
    if t == "new_commitment":
        fields["family"] = same and pred["family"] == truth["family"]
        fields["value"] = same and pred["value"] == truth["value"]
        fields["deadline_step"] = same and pred["deadline_step"] == truth["deadline_step"]
        fields["deadline_phrase_verbatim"] = same and pred["deadline_phrase"].lower() in msg.lower() and \
            resolve_deadline(pred["deadline_phrase"], item["context"]["step"]) == truth["deadline_step"]
        for k, v in truth["spec"].items():
            fields[f"spec.{k}"] = same and fields["family"] and pred["spec"].get(k) == v
        if same and fields["family"]:
            fields["spec.no_extra_keys"] = set(pred["spec"]) == set(truth["spec"])
    elif t == "question":
        fields["spec"] = same and pred["spec"] == truth["spec"]
        for k, v in truth["spec"].items():
            fields[f"spec.{k}"] = same and pred["spec"].get(k) == v
        fields["deadline_step"] = same and pred["deadline_step"] == truth["deadline_step"]
        fields["answer_type"] = same and pred["answer_type"] == truth["answer_type"]
    elif t == "amendment":
        ch = pred.get("changes", {}) if same else {}
        keys = set(truth["diff"]) | {k for k in ch if k != "deadline_phrase"}
        for k in sorted(keys):
            fields[f"changes.{k}"] = same and k in ch and ch[k] == truth["after"].get(k, "<not a field>")
    elif t == "cancellation":
        pass
    elif t == "feedback":
        for k in ("outcome", "step_mentioned", "revealed_answer"):
            fields[k] = same and pred.get(k) == truth[k]
    exact = len(events) == 1 and all(fields.values())
    deadline = None
    if t in ("new_commitment", "question"):
        deadline = fields["deadline_step"]
    elif t == "amendment" and "changes.deadline_step" in fields:
        deadline = fields["changes.deadline_step"]
    return {"exact": exact, "fields": fields, "deadline_ok": deadline, "n_events": len(events),
            "pred_type": pred["type"]}


def score_answer(organ, item: dict, event: dict) -> dict:
    """Compose answers to a question from the oracle memory (offset and bucket forms) and grade them with
    W1's own parser and grader."""
    qa = item["qa"]
    out = {}
    forms = ["offset", "bucket"] if item["truth"]["spec"]["question"] == "last_seen" else ["default"]
    for form in forms:
        mem = dict(qa["memory"])
        if form != "default":
            mem["location_format"] = form
        try:
            text = organ.answer(event, mem)
        except OrganError as e:
            out[form] = {"valid": False, "correct": False, "text": None, "error": str(e)}
            continue
        clauses = parse(text)
        valid = len(clauses) == 1 and clauses[0]["kind"] == "answer" and clauses[0]["id"] == item["truth"]["id"]
        correct = valid and grade_report(qa["truth"], clauses[0]["payload"], qa["player_pos"], qa["tolerance"])
        out[form] = {"valid": bool(valid), "correct": bool(correct), "text": text}
    return out


def _rate(n, d):
    return None if d == 0 else n / d


def _bump(d: dict, key: str, ok: bool):
    c = d.setdefault(key, [0, 0])
    c[0] += int(bool(ok))
    c[1] += 1


def _rates(d: dict) -> dict:
    return {k: {"correct": v[0], "total": v[1], "accuracy": _rate(*v)} for k, v in sorted(d.items())}


def _lat(xs: list) -> dict:
    if not xs:
        return {"n": 0}
    s = sorted(xs)
    return {"n": len(s), "mean_s": statistics.fmean(s), "p50_s": s[len(s) // 2],
            "p95_s": s[min(len(s) - 1, int(0.95 * len(s)))], "max_s": s[-1], "total_s": sum(s)}


def evaluate(organ, corpus: dict, max_failures: int = 40, progress=None) -> dict:
    """Replay the corpus in order (per seed) through ``organ`` and score it."""
    by_type, by_group, by_field, deadline, ans = {}, {}, {}, {}, {}
    latencies, failures, attempts = [], [], {}
    commitments: dict = {}
    items = corpus["items"]
    for i, item in enumerate(items):
        own = commitments.setdefault(item["seed"], {})
        ctx = dict(item["context"])
        ctx["commitments"] = own
        t0 = time.perf_counter()
        events = organ.interpret(item["message"], ctx)
        latencies.append(time.perf_counter() - t0)
        for ev in events:
            if ev["type"] == "new_commitment":
                own[ev["id"]] = ev
            a = ev["provenance"]["attempts"]
            attempts[str(a)] = attempts.get(str(a), 0) + 1
        sc = score_item(item, events)
        t = item["truth"]["type"]
        _bump(by_type, t, sc["exact"])
        _bump(by_group, f"{t}/{group_of(item['truth'])}", sc["exact"])
        for f, ok in sc["fields"].items():
            _bump(by_field, f"{t}.{f}", ok)
        if sc["deadline_ok"] is not None:
            _bump(deadline, t, sc["deadline_ok"])
        if t == "question" and "qa" in item:
            if events and events[0]["type"] == "question":
                res = score_answer(organ, item, events[0])
            else:
                res = {"default": {"valid": False, "correct": False, "text": None, "error": "no question event"}}
            for form, r in res.items():
                key = f"{item['truth']['spec']['question']}/{form}"
                _bump(ans.setdefault("valid", {}), key, r["valid"])
                _bump(ans.setdefault("correct_with_oracle_memory", {}), key, r["correct"])
        if not sc["exact"] and len(failures) < max_failures:
            failures.append({"seed": item["seed"], "step": item["step"], "message": item["message"],
                             "truth": item["truth"], "predicted": events,
                             "wrong_fields": [f for f, ok in sc["fields"].items() if not ok]})
        if progress and (i + 1) % progress == 0:
            done = sum(v[0] for v in by_type.values())
            print(f"  {i + 1}/{len(items)} messages, exact so far {done}/{i + 1}", flush=True)
    n_ok = sum(v[0] for v in by_type.values())
    dl_ok, dl_n = sum(v[0] for v in deadline.values()), sum(v[1] for v in deadline.values())
    av = ans.get("valid", {})
    ac = ans.get("correct_with_oracle_memory", {})
    return {
        "messages": len(items),
        "exact_match": {"correct": n_ok, "total": len(items), "accuracy": _rate(n_ok, len(items))},
        "exact_match_by_event_type": _rates(by_type),
        "exact_match_by_type_and_family": _rates(by_group),
        "per_field_accuracy": _rates(by_field),
        "deadline_step_accuracy": {"correct": dl_ok, "total": dl_n, "accuracy": _rate(dl_ok, dl_n),
                                   "by_event_type": _rates(deadline)},
        "answers": {
            "grammar_valid": {"correct": sum(v[0] for v in av.values()), "total": sum(v[1] for v in av.values()),
                              "by_question_and_form": _rates(av)},
            "correct_with_oracle_memory": {"correct": sum(v[0] for v in ac.values()),
                                           "total": sum(v[1] for v in ac.values()), "by_question_and_form": _rates(ac)},
        },
        "attempts_per_event": attempts,
        "interpret_latency": _lat(latencies),
        "failures_shown": failures,
    }


def corpus_summary(corpus: dict) -> dict:
    counts: dict = {}
    for it in corpus["items"]:
        k = f"{it['truth']['type']}/{group_of(it['truth'])}"
        counts[k] = counts.get(k, 0) + 1
    distinct = len({it["message"] for it in corpus["items"]})
    return {"items": len(corpus["items"]), "distinct_messages": distinct, "by_type_and_family": dict(sorted(counts.items())),
            "seeds": corpus["seeds"], "seed_family": corpus["seed_family"], "episode_length": corpus["episode_length"],
            "actor": corpus["actor"], "config_overrides": corpus["config_overrides"],
            "build_seconds": corpus.get("build_seconds")}
