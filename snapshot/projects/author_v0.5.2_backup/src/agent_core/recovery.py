from __future__ import annotations

from dataclasses import dataclass

from .models import Action, ActionOutcome, ActionKind, WorldState


@dataclass
class RecoveryDecision:
    actions: list[Action]
    reason: str


class RecoveryManager:
    def __init__(self, max_retries_per_action: int = 2):
        self.max_retries_per_action = max_retries_per_action

    def classify(self, action: Action, outcome: ActionOutcome, world: WorldState) -> RecoveryDecision:
        error = (outcome.error or world.last_error or "").lower()

        if action.kind == ActionKind.NAVIGATE_TO.value:
            target = action.params.get("objectId")
            return RecoveryDecision(
                actions=[
                    Action(ActionKind.ROTATE_RIGHT.value, {}, "Refresh view after navigation failure."),
                    Action(ActionKind.NAVIGATE_TO.value, dict(action.params), f"Retry navigation to {target}."),
                ],
                reason="Re-query reachable positions and retry from the updated pose.",
            )

        if action.kind in {
            ActionKind.PICKUP.value,
            ActionKind.PUT.value,
            ActionKind.OPEN.value,
            ActionKind.CLOSE.value,
        }:
            object_id = action.params.get("objectId")
            if any(key in error for key in ["distance", "far", "visible", "view", "reachable", "obstruct", "verification"]):
                return RecoveryDecision(
                    actions=[
                        Action(
                            ActionKind.NAVIGATE_TO.value,
                            {"objectId": object_id},
                            "Reposition to the interaction target after failure.",
                        ),
                        Action(ActionKind.LOOK_DOWN.value, {}, "Check lower part of the interaction view."),
                    ],
                    reason="Interaction failed; reposition and refresh the camera angle.",
                )

        if action.kind in {
            ActionKind.MOVE_AHEAD.value,
            ActionKind.MOVE_BACK.value,
            ActionKind.MOVE_LEFT.value,
            ActionKind.MOVE_RIGHT.value,
        }:
            return RecoveryDecision(
                actions=[Action(ActionKind.ROTATE_RIGHT.value, {}, "Change heading after blocked movement.")],
                reason="Movement failed; change heading before replanning.",
            )

        return RecoveryDecision(
            actions=[Action(ActionKind.ROTATE_RIGHT.value, {}, "Generic recovery sweep.")],
            reason="Generic recovery action.",
        )
