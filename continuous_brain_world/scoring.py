"""Episode and benchmark scoring for W1.

Official Crafter score (Hafner 2021, "Benchmarking the Spectrum of Agent Capabilities", eq. 1)::

    S = exp( (1/N) * sum_i ln(1 + s_i) ) - 1        s_i = success rate of achievement i in percent, N = 22

where the success rate of an achievement is the percentage of episodes in which it was unlocked at least once.
It lies in [0, 100].  ``crafter_score`` implements exactly this; ``success_rates`` builds the s_i.

W1 components (all reported separately -- the summary index is a convenience, not a finding):

* ``obligations.on_time_rate``           fulfilled / (fulfilled + failed), non-report families pooled;
* ``obligations.value_weighted_rate``    same, each obligation weighted by its value (1..3);
* ``obligations.per_family``             the same two rates per family, plus issued/cancelled/amended/unresolved;
* ``memory.accuracy``                    correct answers / report questions resolved (timeouts count wrong);
* ``survival``                           steps survived, fraction of the episode length, died or not.

``W1 index = 100 * mean(C/100, O_v, M, S)`` with ``C`` the official Crafter score over the episode set, ``O_v``
the pooled value-weighted on-time rate, ``M`` the pooled memory accuracy and ``S`` the mean survival fraction.
Equal weights, no tuning; a component with no data (e.g. no questions asked) is dropped from the mean and the
index says so in ``index_components``.
"""
from __future__ import annotations

import math

from crafter import constants

from .obligations import FAMILIES

ACHIEVEMENTS = tuple(constants.achievements)


def crafter_score(success_rates_percent) -> float:
    rates = list(success_rates_percent)
    if len(rates) == 0:
        raise ValueError("no achievements")
    return math.exp(sum(math.log(1.0 + r) for r in rates) / len(rates)) - 1.0


def success_rates(unlocked_per_episode: list) -> dict:
    n = len(unlocked_per_episode)
    return {a: 100.0 * sum(a in set(u) for u in unlocked_per_episode) / n for a in ACHIEVEMENTS}


def _rate(num, den):
    return None if den == 0 else num / den


def episode_summary(layer_state: dict, achievements: dict, episode_length: int) -> dict:
    obs = layer_state["obligations"]
    fam_stats = {}
    for fam in FAMILIES:
        mine = [o for o in obs if o["family"] == fam]
        ful = [o for o in mine if o["status"] == "fulfilled"]
        fail = [o for o in mine if o["status"] == "failed"]
        fam_stats[fam] = {
            "issued": len(mine), "fulfilled": len(ful), "failed": len(fail),
            "cancelled": sum(o["status"] == "cancelled" for o in mine),
            "unresolved": sum(o["status"] in ("open", "amended") for o in mine),
            "amended": sum(any(h["event"] == "amended" for h in o["history"]) for o in mine),
            "value_fulfilled": sum(o["value"] for o in ful), "value_resolved": sum(o["value"] for o in ful + fail),
            "fail_reasons": _count(o["reason"] for o in fail),
        }
        fam_stats[fam]["on_time_rate"] = _rate(len(ful), len(ful) + len(fail))
        fam_stats[fam]["value_weighted_rate"] = _rate(fam_stats[fam]["value_fulfilled"], fam_stats[fam]["value_resolved"])
    non_report = [f for f in FAMILIES if f != "report"]
    ful = sum(fam_stats[f]["fulfilled"] for f in non_report)
    fail = sum(fam_stats[f]["failed"] for f in non_report)
    vf = sum(fam_stats[f]["value_fulfilled"] for f in non_report)
    vr = sum(fam_stats[f]["value_resolved"] for f in non_report)
    rep = [o for o in obs if o["family"] == "report" and o["status"] in ("fulfilled", "failed")]
    per_q = {}
    for o in rep:
        q = o["spec"]["question"]
        d = per_q.setdefault(q, {"asked": 0, "correct": 0, "unanswered": 0})
        d["asked"] += 1
        d["correct"] += o["status"] == "fulfilled"
        d["unanswered"] += o["reason"] == "no_answer"
    steps = int(layer_state["step"])
    unlocked = sorted(a for a, c in achievements.items() if c > 0)
    return {
        "steps": steps, "died": layer_state["death_step"] is not None,
        "survival_fraction": min(1.0, steps / episode_length),
        "crafter": {"unlocked": unlocked, "n_unlocked": len(unlocked),
                    "episode_score": crafter_score([100.0 if a in unlocked else 0.0 for a in ACHIEVEMENTS]),
                    "return": layer_state["crafter_return"]},
        "obligations": {"fulfilled": ful, "failed": fail, "on_time_rate": _rate(ful, ful + fail),
                        "value_fulfilled": vf, "value_resolved": vr, "value_weighted_rate": _rate(vf, vr),
                        "per_family": fam_stats},
        "memory": {"asked": len(rep), "correct": sum(o["status"] == "fulfilled" for o in rep),
                   "accuracy": _rate(sum(o["status"] == "fulfilled" for o in rep), len(rep)),
                   "per_question": per_q},
        "damage": dict(layer_state["stats"]),
    }


def _count(it):
    out = {}
    for x in it:
        out[x] = out.get(x, 0) + 1
    return out


def aggregate(summaries: list) -> dict:
    """Pool episode summaries into benchmark numbers (success rates across episodes, pooled rates)."""
    n = len(summaries)
    rates = success_rates([s["crafter"]["unlocked"] for s in summaries])
    C = crafter_score(rates.values())
    fams = {}
    for fam in FAMILIES:
        tot = {k: sum(s["obligations"]["per_family"][fam][k] for s in summaries)
               for k in ("issued", "fulfilled", "failed", "cancelled", "unresolved", "amended",
                         "value_fulfilled", "value_resolved")}
        tot["on_time_rate"] = _rate(tot["fulfilled"], tot["fulfilled"] + tot["failed"])
        tot["value_weighted_rate"] = _rate(tot["value_fulfilled"], tot["value_resolved"])
        reasons = {}
        for s in summaries:
            for r, c in s["obligations"]["per_family"][fam]["fail_reasons"].items():
                reasons[r] = reasons.get(r, 0) + c
        tot["fail_reasons"] = reasons
        fams[fam] = tot
    ful = sum(s["obligations"]["fulfilled"] for s in summaries)
    fail = sum(s["obligations"]["failed"] for s in summaries)
    vf = sum(s["obligations"]["value_fulfilled"] for s in summaries)
    vr = sum(s["obligations"]["value_resolved"] for s in summaries)
    asked = sum(s["memory"]["asked"] for s in summaries)
    correct = sum(s["memory"]["correct"] for s in summaries)
    per_q = {}
    for s in summaries:
        for q, d in s["memory"]["per_question"].items():
            t = per_q.setdefault(q, {"asked": 0, "correct": 0, "unanswered": 0})
            for k in t:
                t[k] += d[k]
    for d in per_q.values():
        d["accuracy"] = _rate(d["correct"], d["asked"])
    surv = [s["survival_fraction"] for s in summaries]
    O_v, M, S = _rate(vf, vr), _rate(correct, asked), sum(surv) / n
    comps = {"crafter_score/100": C / 100.0, "value_weighted_on_time": O_v, "memory_accuracy": M,
             "survival_fraction": S}
    used = {k: v for k, v in comps.items() if v is not None}
    return {
        "episodes": n,
        "crafter": {"score": C, "success_rates": rates,
                    "mean_unlocked": sum(s["crafter"]["n_unlocked"] for s in summaries) / n,
                    "mean_return": sum(s["crafter"]["return"] for s in summaries) / n},
        "obligations": {"fulfilled": ful, "failed": fail, "on_time_rate": _rate(ful, ful + fail),
                        "value_weighted_rate": O_v, "per_family": fams},
        "memory": {"asked": asked, "correct": correct, "accuracy": M, "per_question": per_q},
        "survival": {"mean_steps": sum(s["steps"] for s in summaries) / n, "mean_fraction": S,
                     "deaths": sum(s["died"] for s in summaries)},
        "w1_index": 100.0 * sum(used.values()) / len(used),
        "index_components": comps,
    }
