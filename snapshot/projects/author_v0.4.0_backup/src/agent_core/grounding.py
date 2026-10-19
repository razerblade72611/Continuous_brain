from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import re
from typing import Iterable

from .models import Action, ActionKind, ObjectState, WorldState


class GroundingError(ValueError):
    """A semantic target could not be translated into a valid simulator target."""


@dataclass(frozen=True)
class GroundedAction:
    action: Action
    semantic_target: str | None = None


class SemanticGrounder:
    """
    Hidden translation layer between the LLM's semantic language and AI2-THOR IDs.

    The grounder may inspect full simulator metadata, but its candidate catalog is
    never sent to the LLM. The model first chooses a semantic reference such as
    "the red apple I can see"; only then does this class resolve the reference.
    """

    def __init__(self, interaction_distance: float = 1.25):
        self.interaction_distance = interaction_distance
        self.last_object_id: str | None = None
        self.last_semantic_target: str | None = None

    @staticmethod
    def _canon(text: str | None) -> str:
        return re.sub(r"[^a-z0-9]+", "", (text or "").lower())

    @staticmethod
    def _tokens(text: str | None) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", (text or "").lower()))

    @staticmethod
    def _eligible(obj: ObjectState, purpose: str) -> bool:
        if purpose == ActionKind.PICKUP.value:
            return obj.pickupable
        if purpose in {ActionKind.OPEN.value, ActionKind.CLOSE.value}:
            return obj.openable
        if purpose == ActionKind.PUT.value:
            return obj.receptacle
        return True

    def _score(self, query: str, obj: ObjectState) -> float:
        q = self._canon(query)
        if not q:
            return -1.0
        typ = self._canon(obj.object_type)
        name = self._canon(obj.name)
        score = 0.0
        if q == typ or q == name:
            score += 100.0
        if q in typ or typ in q:
            score += 45.0
        if q in name or name in q:
            score += 35.0
        score += 20.0 * max(SequenceMatcher(None, q, typ).ratio(), SequenceMatcher(None, q, name).ratio())
        qtok = self._tokens(query)
        otok = self._tokens(obj.object_type) | self._tokens(obj.name)
        if qtok and otok:
            score += 15.0 * (len(qtok & otok) / len(qtok))
        if obj.visible:
            score += 10.0
        if obj.distance is not None:
            score += max(0.0, 6.0 - min(float(obj.distance), 6.0))
        return score

    def resolve(self, target: str, world: WorldState, purpose: str) -> ObjectState:
        phrase = (target or "").strip()
        if not phrase:
            raise GroundingError("The action needs a semantic target phrase.")

        if self._canon(phrase) in {"it", "that", "thatobject", "this", "thisobject"}:
            if self.last_object_id and self.last_object_id in world.objects:
                obj = world.objects[self.last_object_id]
                if self._eligible(obj, purpose):
                    return obj
            raise GroundingError("The reference is ambiguous because no previous object is grounded.")

        eligible = [o for o in world.objects.values() if o.visible and self._eligible(o, purpose)]
        if not eligible:
            raise GroundingError("No currently visible simulator object can ground that semantic reference.")

        ranked = sorted(((self._score(phrase, o), o) for o in eligible), key=lambda row: row[0], reverse=True)
        best_score, best = ranked[0]
        # Deliberately conservative: do not make weak guesses just to make an action happen.
        if best_score < 25.0:
            raise GroundingError(f"Could not confidently ground the semantic target '{phrase}'.")
        return best

    def ground(self, kind: str, target: str | None, params: dict, world: WorldState, reason: str = "") -> GroundedAction:
        primitive = {
            ActionKind.MOVE_AHEAD.value,
            ActionKind.MOVE_BACK.value,
            ActionKind.MOVE_LEFT.value,
            ActionKind.MOVE_RIGHT.value,
            ActionKind.ROTATE_LEFT.value,
            ActionKind.ROTATE_RIGHT.value,
            ActionKind.LOOK_UP.value,
            ActionKind.LOOK_DOWN.value,
            ActionKind.WAIT.value,
        }
        if kind in primitive:
            return GroundedAction(Action(kind, dict(params or {}), reason), None)

        if kind == "Approach":
            obj = self.resolve(str(target or ""), world, ActionKind.NAVIGATE_TO.value)
            self.last_object_id = obj.object_id
            self.last_semantic_target = str(target)
            return GroundedAction(
                Action(
                    ActionKind.NAVIGATE_TO.value,
                    {"objectId": obj.object_id, "interactionDistance": self.interaction_distance},
                    reason,
                ),
                str(target),
            )

        if kind in {
            ActionKind.PICKUP.value,
            ActionKind.PUT.value,
            ActionKind.OPEN.value,
            ActionKind.CLOSE.value,
        }:
            obj = self.resolve(str(target or ""), world, kind)
            self.last_object_id = obj.object_id
            self.last_semantic_target = str(target)
            return GroundedAction(Action(kind, {"objectId": obj.object_id}, reason), str(target))

        raise GroundingError(f"Unsupported embodied action: {kind}")
