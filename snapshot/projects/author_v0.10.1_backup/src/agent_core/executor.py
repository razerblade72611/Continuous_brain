from __future__ import annotations

import json
from pathlib import Path
import math

from .memory import EpisodicMemory
from .models import Action, ActionOutcome, ActionKind, WorldState
from .recovery import RecoveryManager
from .thor_adapter import ThorAdapter


def _object_state(world: WorldState, object_id: str | None):
    if not object_id:
        return None
    return world.objects.get(object_id)


class Verifier:
    def __init__(self, default_interaction_distance: float = 1.5):
        self.default_interaction_distance = default_interaction_distance

    def verify(self, before: WorldState, after: WorldState, action: Action) -> tuple[bool, dict]:
        if action.kind == ActionKind.NAVIGATE_TO.value:
            oid = action.params.get("objectId")
            obj = _object_state(after, oid)
            # v0.10.1 NavigateTo resolves a simulator-authoritative interaction/visibility
            # pose. Distance alone is no longer sufficient evidence of a good approach.
            max_distance = float(action.params.get("interactionDistance", self.default_interaction_distance)) + 0.30
            distance = None if obj is None else obj.distance
            ok = bool(obj and obj.visible and distance is not None and distance <= max_distance)
            return ok, {
                "objectId": oid,
                "distance": distance,
                "max_distance": max_distance,
                "visible": None if obj is None else obj.visible,
                "pose_aware": True,
            }

        if action.kind == ActionKind.CROSS_THRESHOLD.value:
            oid = action.params.get("objectId")
            door_before = _object_state(before, oid)
            door_after = _object_state(after, oid)
            if door_before is None or door_before.position is None:
                return False, {"objectId": oid, "crossed": False, "reason": "door_missing"}
            bx, bz = float(before.pose.position["x"]), float(before.pose.position["z"])
            dx = float(door_before.position["x"]) - bx
            dz = float(door_before.position["z"]) - bz
            norm = math.hypot(dx, dz)
            moved = math.hypot(float(after.pose.position["x"]) - bx, float(after.pose.position["z"]) - bz)
            if norm < 1e-6:
                return False, {"objectId": oid, "crossed": False, "distance_moved": round(moved, 3)}
            ux, uz = dx / norm, dz / norm
            ax = float(after.pose.position["x"]) - float(door_before.position["x"])
            az = float(after.pose.position["z"]) - float(door_before.position["z"])
            beyond = ax * ux + az * uz
            open_ok = bool(door_after and (not door_after.openable or door_after.is_open is True))
            ok = bool(open_ok and moved >= 0.35 and beyond >= 0.20)
            return ok, {
                "objectId": oid,
                "crossed": ok,
                "distance_moved": round(moved, 3),
                "beyond_door_m": round(beyond, 3),
                "door_open": None if door_after is None else door_after.is_open,
            }

        if action.kind == ActionKind.FACE_TARGET.value:
            oid = action.params.get("objectId")
            obj = _object_state(after, oid)
            if obj is None or obj.position is None:
                return False, {"faced": False}
            dx = float(obj.position["x"]) - float(after.pose.position["x"])
            dz = float(obj.position["z"]) - float(after.pose.position["z"])
            desired = math.degrees(math.atan2(dx, dz)) % 360.0
            current = float(after.pose.rotation.get("y", 0.0)) % 360.0
            error = abs((desired - current + 180.0) % 360.0 - 180.0)
            ok = error <= 45.0 + 1e-6
            return ok, {"faced": ok, "yaw_error_degrees": round(error, 2)}

        if action.kind == ActionKind.PICKUP.value:
            oid = action.params.get("objectId")
            ok = after.held_object_id == oid
            return ok, {"expected_held": oid, "actual_held": after.held_object_id}

        if action.kind == ActionKind.PUT.value:
            target_id = action.params.get("objectId")
            held_before = before.held_object_id
            held_after = after.held_object_id
            source_after = _object_state(after, held_before)
            parents = [] if source_after is None else source_after.parent_receptacles
            ok = bool(
                held_before
                and held_after is None
                and source_after is not None
                and target_id in parents
            )
            return ok, {
                "target_receptacle": target_id,
                "held_before": held_before,
                "held_after": held_after,
                "source_parent_receptacles": parents,
            }

        if action.kind == ActionKind.OPEN.value:
            oid = action.params.get("objectId")
            obj = _object_state(after, oid)
            ok = bool(obj and obj.is_open is True)
            return ok, {"objectId": oid, "is_open": None if obj is None else obj.is_open}

        if action.kind == ActionKind.CLOSE.value:
            oid = action.params.get("objectId")
            obj = _object_state(after, oid)
            ok = bool(obj and obj.is_open is False)
            return ok, {"objectId": oid, "is_open": None if obj is None else obj.is_open}

        if action.kind == ActionKind.TOGGLE_ON.value:
            oid = action.params.get("objectId")
            obj = _object_state(after, oid)
            ok = bool(obj and obj.is_toggled is True)
            return ok, {"objectId": oid, "is_toggled": None if obj is None else obj.is_toggled}

        if action.kind == ActionKind.TOGGLE_OFF.value:
            oid = action.params.get("objectId")
            obj = _object_state(after, oid)
            ok = bool(obj and obj.is_toggled is False)
            return ok, {"objectId": oid, "is_toggled": None if obj is None else obj.is_toggled}

        if action.kind == ActionKind.DROP.value:
            held_before = before.held_object_id
            ok = bool(held_before and after.held_object_id is None)
            return ok, {"held_before": held_before, "held_after": after.held_object_id}

        ok = after.last_action_success is not False
        return ok, {"last_action_success": after.last_action_success}


class ActionExecutor:
    def __init__(
        self,
        thor: ThorAdapter,
        memory: EpisodicMemory,
        recovery: RecoveryManager,
        max_steps: int = 80,
        logger_path: str = "runs/actions.jsonl",
    ):
        self.thor = thor
        self.memory = memory
        self.recovery = recovery
        self.max_steps = max_steps
        self.verifier = Verifier(default_interaction_distance=thor.config.visibility_distance)
        self.logger_path = Path(logger_path)
        self.logger_path.parent.mkdir(parents=True, exist_ok=True)
        self.total_steps = 0

    def _log(self, payload: dict):
        with self.logger_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def execute(self, action: Action) -> ActionOutcome:
        remaining = self.max_steps - self.total_steps
        if remaining <= 0:
            raise RuntimeError("Global action budget exhausted.")

        before = self.thor.observe()
        primitive_steps = 0
        try:
            execution = self.thor.execute(action, step_budget=remaining)
            after = execution.state
            primitive_steps = execution.primitive_steps
            self.total_steps += primitive_steps

            event_success = after.last_action_success is not False
            verified, verification = self.verifier.verify(before, after, action)
            success = event_success and verified
            outcome = ActionOutcome(
                action=action,
                success=success,
                error=None if success else (after.last_error or "Verification failed."),
                event_metadata={"lastActionSuccess": after.last_action_success, "auxiliarySkills": list(execution.auxiliary_skills or [])},
                verification=verification,
                attempts=1,
                primitive_steps=primitive_steps,
            )
        except Exception as exc:
            # A thrown exception may happen before Unity consumes a primitive action.
            outcome = ActionOutcome(
                action=action,
                success=False,
                error=f"{type(exc).__name__}: {exc}",
                attempts=1,
                primitive_steps=primitive_steps,
            )

        self._log(outcome.as_dict())
        if outcome.success:
            self.memory.add("action_success", {"action": action.as_dict(), "verification": outcome.verification})
        else:
            signature = f"{action.kind}:{outcome.error or 'unknown'}"
            self.memory.add(
                "action_failure",
                {"action": action.as_dict(), "error": outcome.error, "signature": signature},
            )
        return outcome
