"""r3-T executor lethality analysis (CPU, on the downloaded development eval streams).

For each arm's lives: why it died; where its disagreements with the routine's counterfactual action fall; and which
disagreements are lethal -- conditioned on death within N steps, on damage taken soon after, on threats in view,
and on darkness.  Ranks (routine action) classes by excess disagreement before damage deaths, not by frequency."""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/claude/work/cb")
from continuous_brain_cb2.archive import Archive, fixed  # noqa: E402
from continuous_brain_world import daylight as dl  # noqa: E402
import crafter.constants as cc  # noqa: E402

ACT = list(cc.actions)
ZOMBIE, SKELETON, ARROW = 3, 4, 5
ROOT = Path(sys.argv[1])


def load(stream):
    arch = Archive(ROOT)
    out = defaultdict(list)
    for r in arch.shards():
        if r.stream != stream:
            continue
        t = arch.read(r, ("student", "teacher"),
                      {"student": ["life", "step", "action", "inventory", "done", "dying"],
                       "teacher": ["teacher_action", "teacher_option", "view_object"]})
        s, te = t["student"], t["teacher"]
        out["life"].append(s["life"].to_numpy())
        out["step"].append(s["step"].to_numpy())
        out["action"].append(s["action"].to_numpy())
        out["inv"].append(fixed(s, "inventory", np.int16))
        out["done"].append(s["done"].to_numpy(zero_copy_only=False))
        out["ta"].append(te["teacher_action"].to_numpy())
        out["topt"].append(te["teacher_option"].to_numpy())
        out["vobj"].append(fixed(te, "view_object", np.int8))
    return {k: np.concatenate(v) for k, v in out.items()}


def analyse(d, name, N=30):
    newlife = np.r_[True, (d["life"][1:] != d["life"][:-1]) | (d["step"][1:] <= d["step"][:-1])]
    d["life"] = np.cumsum(newlife)
    lives = np.unique(d["life"])
    n = len(d["life"])
    agree = d["action"] == d["ta"]
    health = d["inv"][:, 0].astype(int)
    food, drink, energy = d["inv"][:, 1], d["inv"][:, 2], d["inv"][:, 3]
    # threat in view: zombie or skeleton (or arrow) anywhere in the 7x9 view; adjacent zombie: within the 3x3 core
    v = d["vobj"].reshape(n, 7, 9) if d["vobj"].shape[1] == 63 else d["vobj"].reshape(n, 9, 7)
    threat = np.isin(v, (ZOMBIE, SKELETON, ARROW)).any((1, 2))
    dark = np.array([dl.is_dark(int(s)) for s in d["step"]])
    # damage next step (health drop not explained by needs at 0)
    nxt_h = np.r_[health[1:], health[-1]]
    same = np.r_[d["life"][1:] == d["life"][:-1], False]
    hurt = same & (nxt_h < health) & (food > 0) & (drink > 0) & (energy > 0)
    endings, near_death, dmg_death = Counter(), np.zeros(n, bool), np.zeros(n, bool)
    for L in lives:
        idx = np.flatnonzero(d["life"] == L)
        last = idx[-1]
        died = bool(d["done"][last]) and len(idx) < 10000 and health[last] <= 9
        if not died:
            endings["survived"] += 1
            continue
        f, dr, e = food[last], drink[last], energy[last]
        cause = "food 0" if f == 0 else "drink 0" if dr == 0 else "energy 0" if e == 0 else "damage"
        endings[cause] += 1
        w = idx[-N:]
        near_death[w] = True
        if cause == "damage":
            dmg_death[w] = True
    rep = {"arm": name, "lives": int(len(lives)), "steps": int(n), "endings": dict(endings),
           "agreement": round(float(agree.mean()), 3),
           "agreement_threat_in_view": round(float(agree[threat].mean()), 3), "share_threat": round(float(threat.mean()), 3),
           "agreement_no_threat": round(float(agree[~threat].mean()), 3),
           "agreement_dark": round(float(agree[dark].mean()), 3), "agreement_light": round(float(agree[~dark].mean()), 3),
           f"agreement_last{N}_before_damage_death": round(float(agree[dmg_death].mean()), 3) if dmg_death.any() else None,
           "hurt_per_100_steps": round(100 * float(hurt.mean()), 2),
           "hurt_per_100_steps_threat": round(100 * float(hurt[threat].mean()), 2),
           "hurt_per_100_steps_dark": round(100 * float(hurt[dark].mean()), 2)}
    # per routine action: count, agreement, agreement before damage deaths, disagreement followed by damage within 5
    hurt_soon = np.zeros(n, bool)
    for k in range(1, 6):
        hurt_soon[:-k] |= hurt[k:] & np.r_[d["life"][k:] == d["life"][:-k]]
    rows = []
    for a in np.unique(d["ta"]):
        m = d["ta"] == a
        dis = m & ~agree
        rows.append({"routine_action": ACT[a], "n": int(m.sum()), "agreement": round(float(agree[m].mean()), 3),
                     "n_last_before_dmg_death": int((m & dmg_death).sum()),
                     "agreement_last_before_dmg_death": round(float(agree[m & dmg_death].mean()), 3) if (m & dmg_death).any() else None,
                     "disagreements": int(dis.sum()),
                     "hurt_within5_after_disagree": round(float(hurt_soon[dis].mean()), 3) if dis.any() else None,
                     "hurt_within5_after_agree": round(float(hurt_soon[m & agree].mean()), 3) if (m & agree).any() else None,
                     "lethal_disagreements": int((dis & dmg_death).sum())})
    rows.sort(key=lambda r: -r["lethal_disagreements"])
    rep["by_routine_action"] = rows
    # what it did instead, when the routine wanted to fight / place stone, under threat
    instead = {}
    for want in ("do", "place_stone", "move_left", "move_right", "move_up", "move_down", "noop", "sleep"):
        a = ACT.index(want)
        m = (d["ta"] == a) & ~agree & threat
        instead[want] = {ACT[k]: int(c) for k, c in Counter(d["action"][m].tolist()).most_common(4)}
    rep["under_threat_routine_wanted_X_it_did"] = instead
    rep["candidate_action_share_under_threat"] = {ACT[k]: round(c / max(threat.sum(), 1), 3)
                                                  for k, c in Counter(d["action"][threat].tolist()).most_common(6)}
    rep["routine_cf_action_share_under_threat"] = {ACT[k]: round(c / max(threat.sum(), 1), 3)
                                                   for k, c in Counter(d["ta"][threat].tolist()).most_common(6)}
    return rep


out = {}
for stream, name in (("eval-r3t-full", "full"), ("eval-r3t-none", "none"), ("eval-r3t-full-routine", "routine")):
    out[name] = analyse(load(stream), name)
print(json.dumps(out, indent=1))
Path("/home/claude/an/r3t_lethality.json").write_text(json.dumps(out, indent=1) + "\n")
