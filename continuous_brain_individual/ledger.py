"""The commitment ledger: what the individual has been asked, has promised, and has been told about it.

Everything here comes from the language organ's events (``continuous_brain_organs.language``), i.e. from what
the individual heard, never from W1's privileged obligation list.  State::

    {"commitments": {id: {"id", "family", "spec", "deadline", "value", "issue_step", "status", "text",
                          "closed_step", "outcome"}},       # status: open | fulfilled | failed | cancelled | expired
     "questions":   {id: {"event": <question event>, "asked_step", "answered_step"}},
     "requests":    [{"id", "issue_step"}],                 # every task request heard, in order (for memory answers)
     "cancellations": [{"id", "step"}],
     "unparsed":    [{"step", "message"}],                  # things heard but not understood
     "organ_calls": int}                                    # the organ's call counter (kept here, not in Python)

``expired``: the deadline passed without feedback; the individual believes it most likely failed.  ``active()``
gives the open task commitments in the format W1's skill routines expect.
"""
from __future__ import annotations

import copy

TASK_FAMILIES = ("collect", "craft", "place", "maintain", "avoid", "visit")
GRACE = 5           # steps after a deadline before an unconfirmed commitment is treated as expired
VISIT_REACH = 3     # "go and look at": within 3 tiles of the feature (W1's stated rule)


def empty() -> dict:
    return {"commitments": {}, "questions": {}, "requests": [], "cancellations": [], "unparsed": [], "organ_calls": 0}


def apply_events(state: dict | None, events: list[dict], step: int) -> dict:
    st = copy.deepcopy(empty() if state is None else state)
    for e in events:
        t = e["type"]
        if t == "new_commitment":
            st["commitments"][e["id"]] = {"id": e["id"], "family": e["family"], "spec": e["spec"],
                                          "deadline": int(e["deadline_step"]), "value": int(e["value"]),
                                          "issue_step": int(e["step"]), "status": "open", "text": e["message"],
                                          "closed_step": None, "outcome": None}
            st["requests"].append({"id": e["id"], "issue_step": int(e["step"])})
        elif t == "amendment":
            c = st["commitments"].get(e["id"])
            if c is not None:
                ch = dict(e["changes"])
                if "deadline_step" in ch:
                    c["deadline"] = int(ch.pop("deadline_step"))
                ch.pop("deadline_phrase", None)
                c["spec"] = {**c["spec"], **ch}
                if c["status"] == "expired" and c["deadline"] > step:
                    c["status"] = "open"
        elif t == "cancellation":
            c = st["commitments"].get(e["id"])
            if c is not None and c["status"] in ("open", "expired"):
                c["status"], c["closed_step"] = "cancelled", step
            st["cancellations"].append({"id": e["id"], "step": int(step)})
        elif t == "question":
            st["questions"][e["id"]] = {"event": e, "asked_step": int(step), "answered_step": None}
        elif t == "feedback":
            c = st["commitments"].get(e["id"])
            if c is not None:
                c["outcome"] = e["outcome"]
                c["status"] = "fulfilled" if e["outcome"] == "fulfilled" else "failed"
                c["closed_step"] = step
            q = st["questions"].get(e["id"])
            if q is not None:
                q["outcome"] = e["outcome"]
        elif t == "unparsed":
            st["unparsed"].append({"step": int(step), "message": e.get("message", "")})
    for c in st["commitments"].values():
        if c["status"] == "open" and step > c["deadline"] + GRACE:
            c["status"], c["closed_step"] = "expired", step
    return st


def active(state: dict | None, step: int) -> list[dict]:
    """Open task commitments in W1's obligation format (for the skill routines)."""
    if state is None:
        return []
    out = []
    for c in state["commitments"].values():
        if c["status"] == "open" and c["family"] in TASK_FAMILIES and step <= c["deadline"]:
            spec = dict(c["spec"])
            if c["family"] == "visit":
                # what the message gives: a direction and a rounded distance ("about 15 steps"); the individual
                # takes the tile it must reach to be at least that far, less the rounding (conservative)
                n = spec.get("stated_distance")
                spec.setdefault("min_dist", max(7, int(n) - 8) if n else 7)
                spec.setdefault("reach_radius", VISIT_REACH)
            out.append({"id": c["id"], "family": c["family"], "spec": spec, "deadline": c["deadline"],
                        "value": c["value"], "issue_step": c["issue_step"], "status": "open", "text": c["text"],
                        "progress": None})
    return sorted(out, key=lambda o: int(o["id"][1:]))


def pending_questions(state: dict | None) -> list[dict]:
    if state is None:
        return []
    return [q["event"] for q in state["questions"].values() if q["answered_step"] is None]


def summary_vector(state: dict | None) -> list[float]:
    """goal_genesis.active_goal: open commitments per task family, then open questions."""
    if state is None:
        return [0.0] * (len(TASK_FAMILIES) + 1)
    counts = [sum(1 for c in state["commitments"].values() if c["status"] == "open" and c["family"] == f)
              for f in TASK_FAMILIES]
    return [float(x) for x in counts] + [float(len(pending_questions(state)))]
