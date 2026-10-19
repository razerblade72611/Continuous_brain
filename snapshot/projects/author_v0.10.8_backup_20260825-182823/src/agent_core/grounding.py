from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import math
import re

from .models import Action, ActionKind, ObjectState, WorldState
from .navigation import facing_yaw


class GroundingError(ValueError):
    """A semantic target could not be translated into a valid simulator target."""


@dataclass(frozen=True)
class GroundedAction:
    action: Action
    semantic_target: str | None = None


class SemanticGrounder:
    """
    Hidden translation layer between the LLM's own semantic references and AI2-THOR.

    No simulator catalog is exposed to the model. A new semantic reference may only
    ground to an object that is actually represented in the current rendered viewport
    (instance segmentation when available, otherwise AI2-THOR's visible flag).

    Once the model itself has grounded a reference, that one reference may persist as
    semantic focus. This gives ordinary object permanence without exposing unrelated
    objects elsewhere in the scene.
    """

    PRONOUNS = {"it", "that", "thatobject", "this", "thisobject", "theobject"}

    def __init__(self, interaction_distance: float = 1.25):
        self.interaction_distance = float(interaction_distance)
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
        if purpose in {ActionKind.TOGGLE_ON.value, ActionKind.TOGGLE_OFF.value}:
            return obj.toggleable
        if purpose == ActionKind.PUT.value:
            return obj.receptacle
        return True

    @staticmethod
    def _angle_delta(current: float, desired: float) -> float:
        return (desired - current + 180.0) % 360.0 - 180.0

    def _in_viewport(self, obj: ObjectState, world: WorldState) -> bool:
        if world.viewport_tracking_available:
            return obj.object_id in world.viewport_object_ids
        return bool(obj.visible)

    def _score(self, query: str, obj: ObjectState) -> float:
        q = self._canon(query)
        if not q:
            return -1.0
        typ = self._canon(obj.object_type)
        name = self._canon(obj.name)
        score = 0.0
        if q == typ or q == name:
            score += 120.0
        if q in typ or typ in q:
            score += 55.0
        if q in name or name in q:
            score += 40.0
        score += 20.0 * max(SequenceMatcher(None, q, typ).ratio(), SequenceMatcher(None, q, name).ratio())
        qtok = self._tokens(query)
        otok = self._tokens(obj.object_type) | self._tokens(obj.name)
        if qtok and otok:
            score += 18.0 * (len(qtok & otok) / len(qtok))
        if obj.distance is not None:
            score += max(0.0, 6.0 - min(float(obj.distance), 6.0))
        return score

    def _focused_object(self, world: WorldState) -> ObjectState | None:
        if self.last_object_id and self.last_object_id in world.objects:
            return world.objects[self.last_object_id]
        return None

    def _can_reuse_focus(self, phrase: str, world: WorldState, purpose: str) -> ObjectState | None:
        obj = self._focused_object(world)
        if obj is None or not self._eligible(obj, purpose):
            return None
        q = self._canon(phrase)
        same_phrase = bool(self.last_semantic_target and q == self._canon(self.last_semantic_target))
        pronoun = q in self.PRONOUNS
        if not (same_phrase or pronoun):
            return None

        # Locomotion/orientation may use a previously grounded object as persistent focus.
        if purpose in {ActionKind.NAVIGATE_TO.value, ActionKind.FACE_TARGET.value}:
            return obj

        # Direct manipulation remains tied to current perception and reach.
        if not self._in_viewport(obj, world):
            raise GroundingError("The previously grounded focus is not currently in the viewport for direct interaction.")
        if obj.distance is not None and float(obj.distance) > self.interaction_distance + 0.25:
            raise GroundingError("The chosen focus is outside direct interaction distance.")
        return obj

    def resolve(self, target: str, world: WorldState, purpose: str) -> ObjectState:
        phrase = (target or "").strip()
        if not phrase:
            raise GroundingError("The action needs a semantic target phrase.")

        reused = self._can_reuse_focus(phrase, world, purpose)
        if reused is not None:
            return reused

        eligible = [
            obj for obj in world.objects.values()
            if self._in_viewport(obj, world) and self._eligible(obj, purpose)
        ]
        if not eligible:
            raise GroundingError("No object in the current rendered viewport can ground that semantic reference.")

        ranked = sorted(((self._score(phrase, obj), obj) for obj in eligible), key=lambda row: row[0], reverse=True)
        best_score, best = ranked[0]
        # Conservative on weak visual-language matches; exact object classes score far above this.
        if best_score < 30.0:
            raise GroundingError(f"Could not confidently ground the semantic target '{phrase}'.")

        if purpose in {ActionKind.PICKUP.value, ActionKind.PUT.value, ActionKind.OPEN.value, ActionKind.CLOSE.value, ActionKind.TOGGLE_ON.value, ActionKind.TOGGLE_OFF.value}:
            if best.distance is not None and float(best.distance) > self.interaction_distance + 0.25:
                raise GroundingError("The chosen focus is outside direct interaction distance.")
        return best

    def _remember(self, phrase: str, obj: ObjectState) -> None:
        self.last_object_id = obj.object_id
        self.last_semantic_target = phrase.strip()

    def try_focus(self, phrase: str | None, world: WorldState) -> bool:
        """Record model-selected attention if it can be grounded; never raises or causes motion."""
        text = (phrase or "").strip()
        if not text:
            return False
        try:
            obj = self.resolve(text, world, ActionKind.FACE_TARGET.value)
        except GroundingError:
            return False
        self._remember(text, obj)
        return True

    def focused_object(self, world: WorldState) -> ObjectState | None:
        """Return only the object previously selected by the model's semantic focus."""
        return self._focused_object(world)

    def focus_snapshot(self, world: WorldState) -> dict | None:
        """Neutral physical facts about the model's own previously selected focus."""
        obj = self._focused_object(world)
        if obj is None or not self.last_semantic_target:
            return None

        in_view = self._in_viewport(obj, world)
        distance = None if obj.distance is None else round(float(obj.distance), 2)
        reached = distance is not None and distance <= self.interaction_distance + 0.15

        yaw_error = None
        if obj.position is not None:
            desired = facing_yaw(world.pose.position, obj.position)
            yaw_error = round(self._angle_delta(float(world.pose.rotation.get("y", 0.0)), desired), 1)

        affordances = ["Observe"]
        if not reached:
            affordances.append("Approach")
        affordances.append("Face")
        if reached and in_view:
            if obj.pickupable and world.held_object_id is None:
                affordances.append("PickupObject")
            if obj.receptacle and world.held_object_id is not None:
                affordances.append("PutObject")
            if obj.openable:
                if obj.is_open is True:
                    affordances.append("CloseObject")
                elif obj.is_open is False:
                    affordances.append("OpenObject")
            if obj.toggleable:
                if obj.is_toggled is True:
                    affordances.append("ToggleObjectOff")
                elif obj.is_toggled is False:
                    affordances.append("ToggleObjectOn")
        if world.held_object_id is not None:
            affordances.append("DropObject")

        return {
            "semantic_reference": self.last_semantic_target,
            "currently_in_viewport": in_view,
            "distance_m": distance,
            "within_approach_distance": reached,
            "bearing_error_degrees": yaw_error,
            "currently_possible_actions": affordances,
        }

    def ground(self, kind: str, target: str | None, params: dict, world: WorldState, reason: str = "") -> GroundedAction:
        params = dict(params or {})
        params.pop("objectId", None)

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
            return GroundedAction(Action(kind, params, reason), None)

        if kind == "Approach":
            obj = self.resolve(str(target or ""), world, ActionKind.NAVIGATE_TO.value)
            self._remember(str(target), obj)
            if obj.distance is not None and float(obj.distance) <= self.interaction_distance + 0.15:
                raise GroundingError("Approach is already satisfied: the chosen focus is already within approach distance.")
            return GroundedAction(
                Action(
                    ActionKind.NAVIGATE_TO.value,
                    {"objectId": obj.object_id, "interactionDistance": self.interaction_distance},
                    reason,
                ),
                str(target),
            )

        if kind == "Face":
            obj = self.resolve(str(target or ""), world, ActionKind.FACE_TARGET.value)
            self._remember(str(target), obj)
            return GroundedAction(
                Action(ActionKind.FACE_TARGET.value, {"objectId": obj.object_id}, reason),
                str(target),
            )

        if kind == "DropObject":
            if world.held_object_id is None:
                raise GroundingError("DropObject requires an object to be held.")
            return GroundedAction(Action(ActionKind.DROP.value, {}, reason), None)

        if kind in {ActionKind.PICKUP.value, ActionKind.PUT.value, ActionKind.OPEN.value, ActionKind.CLOSE.value, ActionKind.TOGGLE_ON.value, ActionKind.TOGGLE_OFF.value}:
            obj = self.resolve(str(target or ""), world, kind)
            self._remember(str(target), obj)
            return GroundedAction(Action(kind, {"objectId": obj.object_id}, reason), str(target))

        raise GroundingError(f"Unsupported embodied action: {kind}")
