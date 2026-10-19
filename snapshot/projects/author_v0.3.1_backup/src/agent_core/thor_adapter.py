from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import time

from .models import Action, ActionKind
from .navigation import NavigationConfig, path_to_actions, shortest_grid_path
from .world_state import normalize_event


@dataclass
class ThorConfig:
    scene: str = "FloorPlan1"
    grid_size: float = 0.25
    rotate_step_degrees: float = 90.0
    visibility_distance: float = 1.5
    interaction_distance: float = 1.25
    width: int = 640
    height: int = 480
    field_of_view: int = 90
    headless: bool = False
    step_delay_seconds: float = 0.0


@dataclass
class ThorExecution:
    event: Any
    state: Any
    primitive_steps: int


class ThorAdapter:
    """Boundary around AI2-THOR plus deterministic grid navigation."""

    def __init__(self, config: ThorConfig):
        self.config = config
        self.controller = None
        self.step_index = 0
        self.state = None

    def start(self) -> None:
        from ai2thor.controller import Controller

        kwargs = dict(
            agentMode="default",
            visibilityDistance=self.config.visibility_distance,
            scene=self.config.scene,
            gridSize=self.config.grid_size,
            snapToGrid=True,
            rotateStepDegrees=self.config.rotate_step_degrees,
            renderDepthImage=False,
            renderInstanceSegmentation=False,
            width=self.config.width,
            height=self.config.height,
            fieldOfView=self.config.field_of_view,
        )
        if self.config.headless:
            from ai2thor.platform import CloudRendering
            kwargs["platform"] = CloudRendering

        self.controller = Controller(**kwargs)
        self.step_index = 0
        self.state = normalize_event(self.controller.last_event, self.config.scene, self.step_index)

    def close(self) -> None:
        if self.controller is not None:
            self.controller.stop()
            self.controller = None

    def observe(self):
        if self.controller is None:
            raise RuntimeError("THOR adapter is not started.")
        self.state = normalize_event(self.controller.last_event, self.config.scene, self.step_index)
        return self.state

    def get_reachable_positions(self) -> list[dict[str, float]]:
        if self.controller is None:
            raise RuntimeError("THOR adapter is not started.")
        event = self.controller.step(action="GetReachablePositions")
        if event.metadata.get("lastActionSuccess") is False:
            raise RuntimeError(event.metadata.get("errorMessage") or "GetReachablePositions failed.")
        return list(event.metadata.get("actionReturn") or [])

    def _execute_primitive(self, action: Action) -> ThorExecution:
        event = self.controller.step(action=action.kind, **action.params)
        if self.config.step_delay_seconds > 0:
            time.sleep(self.config.step_delay_seconds)
        self.step_index += 1
        self.state = normalize_event(event, self.config.scene, self.step_index)
        return ThorExecution(event=event, state=self.state, primitive_steps=1)

    def _navigate_to(self, action: Action, step_budget: int | None = None) -> ThorExecution:
        object_id = str(action.params.get("objectId", ""))
        if not object_id:
            raise ValueError("NavigateTo requires params.objectId.")

        world = self.observe()
        target = world.objects.get(object_id)
        if target is None or target.position is None:
            raise ValueError(f"NavigateTo target not found or has no position: {object_id}")

        nav_config = NavigationConfig(
            grid_size=self.config.grid_size,
            rotate_step_degrees=self.config.rotate_step_degrees,
            interaction_distance=float(action.params.get("interactionDistance", self.config.interaction_distance)),
        )
        reachable = self.get_reachable_positions()
        path = shortest_grid_path(
            reachable,
            world.pose.position,
            target.position,
            nav_config,
        )
        primitive_actions = path_to_actions(
            path,
            initial_yaw=float(world.pose.rotation.get("y", 0.0)),
            target_position=target.position,
            config=nav_config,
        )

        if not primitive_actions:
            # Already at the best reachable position; keep the current event.
            return ThorExecution(self.controller.last_event, self.observe(), 0)

        if step_budget is not None and len(primitive_actions) > step_budget:
            primitive_actions = primitive_actions[:step_budget]

        last: ThorExecution | None = None
        for primitive in primitive_actions:
            last = self._execute_primitive(primitive)
            if last.state.last_action_success is False:
                break

        if last is None:
            return ThorExecution(self.controller.last_event, self.observe(), 0)
        last.primitive_steps = len(primitive_actions) if last.state.last_action_success is not False else max(1, self.step_index - world.step_index)
        return last

    def execute(self, action: Action, step_budget: int | None = None) -> ThorExecution:
        if self.controller is None:
            raise RuntimeError("THOR adapter is not started.")
        if action.kind == ActionKind.NAVIGATE_TO.value:
            return self._navigate_to(action, step_budget=step_budget)
        return self._execute_primitive(action)

    def raw_event(self):
        return None if self.controller is None else self.controller.last_event
