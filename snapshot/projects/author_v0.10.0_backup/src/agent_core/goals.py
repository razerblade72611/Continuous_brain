from __future__ import annotations

from dataclasses import dataclass
import re


def canonical_name(value: str | None) -> str:
    """Normalize human/object names for tolerant comparisons."""
    if not value:
        return ""
    return re.sub(r"[^a-z0-9]+", "", value.lower())


@dataclass(frozen=True)
class GoalSpec:
    raw: str
    pickup_object: str | None = None
    put_object: str | None = None
    put_target: str | None = None
    put_relation: str | None = None
    open_object: str | None = None
    close_object: str | None = None

    @property
    def has_put(self) -> bool:
        return bool(self.put_target)

    @property
    def source_object(self) -> str | None:
        source = self.put_object or self.pickup_object
        if source and canonical_name(source) in {"it", "them"}:
            return self.pickup_object
        return source


def parse_goal(goal: str) -> GoalSpec:
    """
    Parse the small deterministic task language supported by the fallback planner.

    This intentionally stays conservative. Unknown goals are left for the LLM
    planner or exploration fallback rather than being guessed as complete.
    """
    text = " ".join(goal.strip().split())

    pickup = re.search(
        r"(?:pick\s*up|pickup|get|grab)\s+(?:the\s+)?"
        r"(?P<object>[a-zA-Z0-9_ -]+?)(?=\s+and\b|[.!?]?$)",
        text,
        flags=re.IGNORECASE,
    )

    put = re.search(
        r"(?:put|place)\s+(?:the\s+)?"
        r"(?P<object>[a-zA-Z0-9_ -]+?)\s+"
        r"(?P<relation>on|in|into)\s+(?:the\s+)?"
        r"(?P<target>[a-zA-Z0-9_ -]+?)(?=[.!?]?$)",
        text,
        flags=re.IGNORECASE,
    )

    open_match = re.fullmatch(
        r"open\s+(?:the\s+)?(?P<object>[a-zA-Z0-9_ -]+?)[.!?]?",
        text,
        flags=re.IGNORECASE,
    )
    close_match = re.fullmatch(
        r"close\s+(?:the\s+)?(?P<object>[a-zA-Z0-9_ -]+?)[.!?]?",
        text,
        flags=re.IGNORECASE,
    )

    pickup_object = pickup.group("object").strip() if pickup else None
    put_object = put.group("object").strip() if put else None
    put_target = put.group("target").strip() if put else None
    put_relation = put.group("relation").lower() if put else None

    if put_object and canonical_name(put_object) in {"it", "them"}:
        put_object = pickup_object

    return GoalSpec(
        raw=goal,
        pickup_object=pickup_object,
        put_object=put_object,
        put_target=put_target,
        put_relation=put_relation,
        open_object=open_match.group("object").strip() if open_match else None,
        close_object=close_match.group("object").strip() if close_match else None,
    )
