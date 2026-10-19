"""Bounded context-qualified goal composition; no learned weights or hidden state.

The author sees only a map of context names to autobiographical A/B estimates.
The two-clause grammar, ordering rule, +0.12 target and horizon are engineered.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Mapping, Sequence

CONTEXTS = ("context_0", "context_1", "context_2")
TARGET_DELTA = 0.12
TIE_EPSILON = 1e-8
MAX_EPISODES_PER_CLAUSE = 16
MAX_TOTAL_EPISODES = 32


def checked_pair(values):
    if len(values) != 2:
        raise ValueError("expected exactly two reliability estimates")
    pair = tuple(float(v) for v in values)
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in pair):
        raise ValueError("reliabilities must be finite and within [0, 1]")
    return pair


@dataclass(frozen=True)
class GoalClause:
    context: str
    primary: int
    baseline_reliability: float
    target_reliability: float

    def __post_init__(self):
        if self.context not in CONTEXTS or type(self.primary) is not int or self.primary not in (1, 2):
            raise ValueError("invalid context or primitive")
        b, t = checked_pair((self.baseline_reliability, self.target_reliability))
        if t <= b:
            raise ValueError("target must exceed baseline")


@dataclass(frozen=True)
class CompositeGoal:
    clauses: tuple[GoalClause, GoalClause]

    def __post_init__(self):
        if len(self.clauses) != 2 or len({c.context for c in self.clauses}) != 2:
            raise ValueError("two clauses in different contexts are required")

    def payload(self):
        return {"operator": "THEN", "clauses": [asdict(c) for c in self.clauses]}

    @classmethod
    def restore(cls, raw):
        if raw.get("operator") != "THEN":
            raise ValueError("unsupported goal operator")
        return cls(tuple(GoalClause(**c) for c in raw["clauses"]))


def author_goal(reliabilities: Mapping[str, Sequence[float]]):
    """Pick the two weakest directional commitments, in separate contexts.

    No subject ID, capability, success label, desired context or external target
    is accepted. Context names bind histories to task locations, not priority.
    Exact score ties use lexical context order; this is not learned preference.
    """
    if set(reliabilities) != set(CONTEXTS):
        raise ValueError("all three context estimates are required")
    candidates = []
    for context in CONTEXTS:
        a, b = checked_pair(reliabilities[context])
        if abs(a - b) <= TIE_EPSILON:
            continue
        primary = 1 if a < b else 2
        baseline = min(a, b)
        target = min(1.0, baseline + TARGET_DELTA)
        if target - baseline <= TIE_EPSILON:
            continue
        candidates.append(GoalClause(context, primary, baseline, target))
    candidates.sort(key=lambda c: (c.baseline_reliability, c.context))
    return CompositeGoal(tuple(candidates[:2])) if len(candidates) >= 2 else None


def goal_payload(goal):
    return None if goal is None else goal.payload()


def clause_value(clause, reliabilities):
    return checked_pair(reliabilities[clause.context])[clause.primary - 1]


def targets_met(goal, reliabilities):
    return [clause_value(c, reliabilities) >= c.target_reliability for c in goal.clauses]


@dataclass
class CompositeProgress:
    goal: CompositeGoal
    cursor: int = 0
    attempts: list[int] = field(default_factory=lambda: [0, 0])
    exhausted: bool = False

    def __post_init__(self):
        if type(self.cursor) is not int or not 0 <= self.cursor <= 2:
            raise ValueError("invalid cursor")
        if len(self.attempts) != 2 or any(type(n) is not int or not 0 <= n <= MAX_EPISODES_PER_CLAUSE for n in self.attempts):
            raise ValueError("invalid episode counts")

    @property
    def complete(self):
        return self.cursor == 2 and not self.exhausted

    def next_clause(self, reliabilities):
        if self.exhausted:
            return None
        while self.cursor < 2:
            clause = self.goal.clauses[self.cursor]
            # Check completion BEFORE exhaustion, including the 16th episode.
            if clause_value(clause, reliabilities) >= clause.target_reliability:
                self.cursor += 1
                continue
            if self.attempts[self.cursor] >= MAX_EPISODES_PER_CLAUSE:
                self.exhausted = True
                return None
            return clause
        return None

    def record_episode(self):
        if self.exhausted or self.cursor >= 2:
            raise RuntimeError("no active clause")
        if self.attempts[self.cursor] >= MAX_EPISODES_PER_CLAUSE:
            raise RuntimeError("episode budget exhausted")
        self.attempts[self.cursor] += 1

    def payload(self):
        return {"goal": self.goal.payload(), "cursor": self.cursor,
                "attempts": list(self.attempts), "exhausted": self.exhausted}

    @classmethod
    def restore(cls, raw):
        return cls(CompositeGoal.restore(raw["goal"]), raw["cursor"],
                   list(raw["attempts"]), raw["exhausted"])


def run_composite(backend, progress, trace=None, pause_after=None):
    """Backend executes context/primitive commands and reports autobiography.

    pause_after is an infrastructure checkpoint boundary, never a goal input.
    This function cannot inspect capabilities or evaluator-known fragility.
    """
    trace = [] if trace is None else trace
    while True:
        clause = progress.next_clause(backend.reliabilities())
        if clause is None:
            return trace
        if pause_after is not None and len(trace) >= pause_after:
            return trace
        metrics = backend.execute(clause.context, clause.primary, len(trace))
        trace.append({"clause_index": progress.cursor, "context": clause.context,
                      "primary": clause.primary, "metrics": metrics})
        progress.record_episode()


def run_control(backend, goal, mode):
    """Fixed 32-episode budget; compare final ORIGINAL target predicates.

    drop_second spends all episodes on the first clause. misbind_second routes
    its second 16 episodes to the unselected context with the same primitive.
    These are engineered interventions, not alternative trained agents.
    """
    if mode not in ("drop_second", "misbind_second"):
        raise ValueError(mode)
    spare = next(c for c in CONTEXTS if c not in {g.context for g in goal.clauses})
    trace = []
    for step in range(MAX_TOTAL_EPISODES):
        clause = goal.clauses[0]
        context = clause.context
        if mode == "misbind_second" and step >= MAX_EPISODES_PER_CLAUSE:
            clause = goal.clauses[1]
            context = spare
        metrics = backend.execute(context, clause.primary, step)
        trace.append({"context": context, "primary": clause.primary, "metrics": metrics})
    return trace
